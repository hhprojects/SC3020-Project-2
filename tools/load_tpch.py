"""
load_tpch.py - One-off helper that loads TPC-H .tbl files into PostgreSQL.

What it does:
  1. Drops and re-creates the 8 TPC-H tables (safe to re-run).
  2. Streams each <table>.tbl file into PostgreSQL with COPY, removing the
     trailing '|' that dbgen puts at the end of every line.
  3. Adds primary keys and secondary indexes on join/filter columns.
  4. Runs ANALYZE so the optimizer has up-to-date statistics.

Works on macOS and Windows. Example:
  python load_tpch.py --data-dir path/to/dbgen --dbname tpch
  python load_tpch.py --data-dir C:\\tpch\\dbgen --dbname tpch --user postgres --password secret

This is a setup utility, not part of the four submitted program files.
"""

import argparse
import time
from pathlib import Path

import psycopg2

# Tables in load order (small/parent tables first).
TABLE_DDL = {
    "region": """
        CREATE TABLE region (
            r_regionkey INTEGER      NOT NULL,
            r_name      CHAR(25)     NOT NULL,
            r_comment   VARCHAR(152)
        )""",
    "nation": """
        CREATE TABLE nation (
            n_nationkey INTEGER      NOT NULL,
            n_name      CHAR(25)     NOT NULL,
            n_regionkey INTEGER      NOT NULL,
            n_comment   VARCHAR(152)
        )""",
    "part": """
        CREATE TABLE part (
            p_partkey     INTEGER        NOT NULL,
            p_name        VARCHAR(55)    NOT NULL,
            p_mfgr        CHAR(25)       NOT NULL,
            p_brand       CHAR(10)       NOT NULL,
            p_type        VARCHAR(25)    NOT NULL,
            p_size        INTEGER        NOT NULL,
            p_container   CHAR(10)       NOT NULL,
            p_retailprice DECIMAL(15, 2) NOT NULL,
            p_comment     VARCHAR(23)    NOT NULL
        )""",
    "supplier": """
        CREATE TABLE supplier (
            s_suppkey   INTEGER        NOT NULL,
            s_name      CHAR(25)       NOT NULL,
            s_address   VARCHAR(40)    NOT NULL,
            s_nationkey INTEGER        NOT NULL,
            s_phone     CHAR(15)       NOT NULL,
            s_acctbal   DECIMAL(15, 2) NOT NULL,
            s_comment   VARCHAR(101)   NOT NULL
        )""",
    "partsupp": """
        CREATE TABLE partsupp (
            ps_partkey    INTEGER        NOT NULL,
            ps_suppkey    INTEGER        NOT NULL,
            ps_availqty   INTEGER        NOT NULL,
            ps_supplycost DECIMAL(15, 2) NOT NULL,
            ps_comment    VARCHAR(199)   NOT NULL
        )""",
    "customer": """
        CREATE TABLE customer (
            c_custkey    INTEGER        NOT NULL,
            c_name       VARCHAR(25)    NOT NULL,
            c_address    VARCHAR(40)    NOT NULL,
            c_nationkey  INTEGER        NOT NULL,
            c_phone      CHAR(15)       NOT NULL,
            c_acctbal    DECIMAL(15, 2) NOT NULL,
            c_mktsegment CHAR(10)       NOT NULL,
            c_comment    VARCHAR(117)   NOT NULL
        )""",
    "orders": """
        CREATE TABLE orders (
            o_orderkey      INTEGER        NOT NULL,
            o_custkey       INTEGER        NOT NULL,
            o_orderstatus   CHAR(1)        NOT NULL,
            o_totalprice    DECIMAL(15, 2) NOT NULL,
            o_orderdate     DATE           NOT NULL,
            o_orderpriority CHAR(15)       NOT NULL,
            o_clerk         CHAR(15)       NOT NULL,
            o_shippriority  INTEGER        NOT NULL,
            o_comment       VARCHAR(79)    NOT NULL
        )""",
    "lineitem": """
        CREATE TABLE lineitem (
            l_orderkey      INTEGER        NOT NULL,
            l_partkey       INTEGER        NOT NULL,
            l_suppkey       INTEGER        NOT NULL,
            l_linenumber    INTEGER        NOT NULL,
            l_quantity      DECIMAL(15, 2) NOT NULL,
            l_extendedprice DECIMAL(15, 2) NOT NULL,
            l_discount      DECIMAL(15, 2) NOT NULL,
            l_tax           DECIMAL(15, 2) NOT NULL,
            l_returnflag    CHAR(1)        NOT NULL,
            l_linestatus    CHAR(1)        NOT NULL,
            l_shipdate      DATE           NOT NULL,
            l_commitdate    DATE           NOT NULL,
            l_receiptdate   DATE           NOT NULL,
            l_shipinstruct  CHAR(25)       NOT NULL,
            l_shipmode      CHAR(10)       NOT NULL,
            l_comment       VARCHAR(44)    NOT NULL
        )""",
}

# Added after loading - building indexes on a full table is much faster
# than maintaining them row by row during COPY.
PRIMARY_KEYS = [
    "ALTER TABLE region   ADD PRIMARY KEY (r_regionkey)",
    "ALTER TABLE nation   ADD PRIMARY KEY (n_nationkey)",
    "ALTER TABLE part     ADD PRIMARY KEY (p_partkey)",
    "ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey)",
    "ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey)",
    "ALTER TABLE customer ADD PRIMARY KEY (c_custkey)",
    "ALTER TABLE orders   ADD PRIMARY KEY (o_orderkey)",
    "ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber)",
]

# Indexes on foreign-key and commonly filtered columns, so the optimizer
# has realistic choices between index scans and sequential scans.
SECONDARY_INDEXES = [
    "CREATE INDEX idx_nation_regionkey   ON nation (n_regionkey)",
    "CREATE INDEX idx_supplier_nationkey ON supplier (s_nationkey)",
    "CREATE INDEX idx_customer_nationkey ON customer (c_nationkey)",
    "CREATE INDEX idx_partsupp_suppkey   ON partsupp (ps_suppkey)",
    "CREATE INDEX idx_orders_custkey     ON orders (o_custkey)",
    "CREATE INDEX idx_orders_orderdate   ON orders (o_orderdate)",
    "CREATE INDEX idx_lineitem_partkey   ON lineitem (l_partkey)",
    "CREATE INDEX idx_lineitem_suppkey   ON lineitem (l_suppkey)",
    "CREATE INDEX idx_lineitem_shipdate  ON lineitem (l_shipdate)",
]


class TblReader:
    """File-like wrapper that strips the trailing '|' from each dbgen line.

    psycopg2's copy_expert() only needs a read(size) method, so this lets us
    stream a multi-GB file straight into COPY without writing a cleaned copy.
    """

    def __init__(self, path: Path):
        self._file = open(path, "r", encoding="utf-8", newline="")
        self._buffer = ""

    def read(self, size: int = -1) -> str:
        # Fill the buffer with cleaned lines until we have enough characters.
        while size < 0 or len(self._buffer) < size:
            line = self._file.readline()
            if not line:
                break
            line = line.rstrip("\r\n")
            if line.endswith("|"):
                line = line[:-1]
            self._buffer += line + "\n"

        if size < 0:
            chunk, self._buffer = self._buffer, ""
        else:
            chunk, self._buffer = self._buffer[:size], self._buffer[size:]
        return chunk

    def close(self):
        self._file.close()


def run_timed(cursor, label: str, sql: str):
    """Execute one SQL statement and print how long it took."""
    start = time.time()
    cursor.execute(sql)
    print(f"  {label} ({time.time() - start:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Load TPC-H .tbl files into PostgreSQL.")
    parser.add_argument("--data-dir", required=True, help="Folder containing the 8 .tbl files")
    parser.add_argument("--dbname", default="tpch")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="5432")
    parser.add_argument("--user", default=None, help="Defaults to your OS username (Mac) - use 'postgres' on Windows")
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    missing = [t for t in TABLE_DDL if not (data_dir / f"{t}.tbl").exists()]
    if missing:
        raise SystemExit(f"Missing .tbl files in {data_dir}: {', '.join(missing)}")

    conn = psycopg2.connect(
        dbname=args.dbname, host=args.host, port=args.port,
        user=args.user, password=args.password,
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("1/4 Creating tables...")
    for table in reversed(list(TABLE_DDL)):
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for table, ddl in TABLE_DDL.items():
        cur.execute(ddl)

    print("2/4 Loading data...")
    for table in TABLE_DDL:
        reader = TblReader(data_dir / f"{table}.tbl")
        start = time.time()
        try:
            cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT text, DELIMITER '|')", reader)
        finally:
            reader.close()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]:,} rows ({time.time() - start:.1f}s)")

    print("3/4 Adding primary keys and indexes...")
    for sql in PRIMARY_KEYS + SECONDARY_INDEXES:
        run_timed(cur, sql.split(" ON ")[0] if "INDEX" in sql else sql, sql)

    print("4/4 Running ANALYZE...")
    run_timed(cur, "ANALYZE", "ANALYZE")

    cur.close()
    conn.close()
    print("Done. TPC-H is ready.")


if __name__ == "__main__":
    main()
