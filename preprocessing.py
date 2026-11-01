"""preprocessing.py - Reading inputs and preparing query plans for comparison.

Responsibilities:
  1. Load configuration (database login, AI model) from config.json and the
     API key from .env.
  2. Connect to PostgreSQL and read the schema of whatever database is chosen
     (tables, columns, keys, indexes, row counts).
  3. Retrieve PostgreSQL's query execution plan (QEP) with EXPLAIN (FORMAT JSON)
     and convert it into a PlanNode tree.
  4. Ask the AI agent (Claude, Gemini or OpenAI) for its plan, check the answer
     against the schema and the query (to catch hallucinations), retry if needed,
     and convert it into the same PlanNode tree format, so comparison.py only
     ever deals with one format.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "database": {"host": "localhost", "port": 5432, "dbname": "tpch", "user": "", "password": ""},
    "ai": {"provider": "gemini", "model": "", "max_retries": 2,
           "temperature": None, "effort": "high"},
    "explain": {"statement_timeout_ms": 30000},
}

CONDITION_KEYS = [
    "Hash Cond", "Merge Cond", "Join Filter", "Index Cond",
    "Recheck Cond", "Filter", "TID Cond", "One-Time Filter",
]

ATTRIBUTE_KEYS = [
    "Strategy", "Partial Mode", "Sort Key", "Group Key", "Parallel Aware",
    "Workers Planned", "Parent Relationship", "Subplan Name", "CTE Name",
    "Function Name", "Scan Direction", "Command", "Hash Key",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load config.json, filling any missing entries from DEFAULT_CONFIG.

    Also loads .env so API keys are available.
    """
    load_dotenv(BASE_DIR / ".env")

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for section, values in user_config.items():
            config.setdefault(section, {}).update(values)
    return config


def get_api_key(provider: str) -> Optional[str]:
    """Return the API key for an AI provider from the environment, e.g. CLAUDE_API_KEY."""
    return os.environ.get(f"{provider.upper()}_API_KEY")


@dataclass
class PlanNode:
    """One operator in a query execution plan, e.g. 'Hash Join' or 'Seq Scan'.

    Attributes:
        node_type:      PostgreSQL operator name
        relation:       base table read, for scan nodes
        alias:          alias used in the query, e.g. 'l' in 'lineitem l'
        index_name:     index used, for index scans
        join_type:      Inner / Left / Semi / Anti ..., for joins
        conditions:     predicates, e.g. {'Hash Cond': '(a = b)'}
        attributes:     other EXPLAIN fields, e.g. {'Strategy': 'Hashed'}
        estimated_rows: estimated output rows
        estimated_cost: estimated total cost of this subtree
        children:       input nodes
    """

    node_type: str
    relation: Optional[str] = None
    alias: Optional[str] = None
    index_name: Optional[str] = None
    join_type: Optional[str] = None
    conditions: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    estimated_rows: Optional[float] = None
    estimated_cost: Optional[float] = None
    children: list = field(default_factory=list)

    @classmethod
    def from_explain_json(cls, plan: dict) -> "PlanNode":
        """Recursively build a PlanNode tree from one 'Plan' object of EXPLAIN JSON."""
        return cls(
            node_type=plan.get("Node Type", "Unknown"),
            relation=plan.get("Relation Name"),
            alias=plan.get("Alias"),
            index_name=plan.get("Index Name"),
            join_type=plan.get("Join Type"),
            conditions={k: plan[k] for k in CONDITION_KEYS if k in plan},
            attributes={k: plan[k] for k in ATTRIBUTE_KEYS if k in plan},
            estimated_rows=plan.get("Plan Rows"),
            estimated_cost=plan.get("Total Cost"),
            children=[cls.from_explain_json(child) for child in plan.get("Plans", [])],
        )

    def walk(self) -> Iterator["PlanNode"]:
        """Yield every node in the subtree, parent before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def base_relations(self) -> frozenset:
        """Set of base tables read somewhere in this subtree."""
        return frozenset(n.relation for n in self.walk() if n.relation)

    def size(self) -> int:
        """Number of nodes in the subtree."""
        return sum(1 for _ in self.walk())

    def label(self) -> str:
        """Short one-line description, e.g. 'Index Scan using part_pkey on part'.

        Uses the same prefixes as PostgreSQL's own text EXPLAIN, e.g. 'Parallel Seq Scan'
        or 'Finalize Aggregate'.
        """
        text = self.node_type
        if self.attributes.get("Parallel Aware"):
            text = "Parallel " + text
        if self.attributes.get("Partial Mode") in ("Partial", "Finalize"):
            text = f"{self.attributes['Partial Mode']} {text}"
        if self.join_type and self.join_type != "Inner":
            text += f" ({self.join_type})"
        if "Strategy" in self.attributes and self.node_type == "Aggregate":
            text += f" ({self.attributes['Strategy']})"
        if self.index_name:
            text += f" using {self.index_name}"
        if self.relation:
            text += f" on {self.relation}"
            if self.alias and self.alias != self.relation:
                text += f" {self.alias}"
        return text

    def to_text(self, show_details: bool = True) -> str:
        """Render the subtree as an indented text tree (useful for debugging/printing)."""
        return "\n".join(text for _, text in self.to_lines(show_details))

    def to_lines(self, show_details: bool = True) -> list:
        """Render the subtree as (node, line of text) pairs, so a GUI can style each node's lines."""
        lines = []
        self._render(lines, prefix="", is_last=True, is_root=True, show_details=show_details)
        return lines

    def _render(self, lines, prefix, is_last, is_root, show_details):
        """Append this node's line(s) to `lines`, then its children's, drawing the tree branches."""
        connector = "" if is_root else ("└── " if is_last else "├── ")
        line = prefix + connector + self.label()
        if show_details:
            estimates = []
            if self.estimated_rows is not None:
                estimates.append(f"rows={self.estimated_rows:,.0f}")
            if self.estimated_cost is not None:
                estimates.append(f"cost={self.estimated_cost:,.2f}")
            if estimates:
                line += f"  [{', '.join(estimates)}]"
        lines.append((self, line))

        child_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
        if show_details:
            for key, value in self.conditions.items():
                lines.append((self, f"{child_prefix}{'│   ' if self.children else '    '}{key}: {value}"))
        for i, child in enumerate(self.children):
            child._render(lines, child_prefix, i == len(self.children) - 1, False, show_details)


@dataclass
class QueryPlan:
    """A full plan for one query, plus where it came from.

    Attributes:
        root:             the PlanNode tree
        source:           'PostgreSQL' or the AI model name
        raw:              the original JSON, kept for display and debugging
        reasoning:        the AI's own short justification of its plan
        warnings:         problems found while validating an AI plan
        serious_warnings: the subset that suggests hallucination
        attempts:         AI calls made (more than 1 means the answer was retried)
    """

    root: PlanNode
    source: str
    raw: object = None
    reasoning: str = ""
    warnings: list = field(default_factory=list)
    serious_warnings: list = field(default_factory=list)
    attempts: int = 1


@dataclass
class TableInfo:
    """One table of the schema.

    Attributes:
        row_estimate: estimated rows (None if the table has never been analysed)
        pages:        size on disk in pages
        columns:      (name, type, not_null) tuples
        constraints:  (kind, name, definition) tuples - primary, foreign and unique keys
        indexes:      (name, definition) tuples
    """
    schema: str
    name: str
    row_estimate: Optional[int]
    pages: int
    columns: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    indexes: list = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """Table name, prefixed with its schema unless it is in 'public'."""
        return self.name if self.schema == "public" else f"{self.schema}.{self.name}"


@dataclass
class SchemaInfo:
    """Everything we know about the database that a planner would care about.

    Tables (TableInfo), the server version, and planner settings such as work_mem and
    random_page_cost.
    """

    database: str
    server_version: str
    tables: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Compact plain-text description of the schema, used in the AI prompt."""
        lines = [f"Database: {self.database} (PostgreSQL {self.server_version})", ""]
        for t in self.tables:
            rows = f"{t.row_estimate:,}" if t.row_estimate is not None else "unknown"
            lines.append(f"Table {t.qualified_name} (~{rows} rows, {t.pages:,} pages)")
            for col_name, col_type, not_null in t.columns:
                lines.append(f"  - {col_name} {col_type}{' NOT NULL' if not_null else ''}")
            for kind, _, definition in t.constraints:
                lines.append(f"  {kind}: {definition}")
            for idx_name, definition in t.indexes:
                lines.append(f"  Index {idx_name}: {definition}")
            lines.append("")
        if self.settings:
            lines.append("Planner settings: " + ", ".join(f"{k}={v}" for k, v in self.settings.items()))
        return "\n".join(lines)


class PlanRetrievalError(Exception):
    """Raised with a user-friendly message when a plan cannot be produced."""


_USER_SCHEMAS = "n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname !~ '^pg_toast'"

_PLANNER_SETTINGS = [
    "work_mem", "shared_buffers", "effective_cache_size", "random_page_cost",
    "seq_page_cost", "max_parallel_workers_per_gather",
]


class DatabaseConnection:
    """Thin wrapper around a read-only psycopg2 connection."""

    def __init__(self, host="localhost", port=5432, dbname="tpch", user="", password="",
                 statement_timeout_ms=30000):
        """Store the connection settings.

        An empty user or password lets PostgreSQL use its defaults (your OS username on macOS).
        """
        self.params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user or None,
            "password": password or None,
            "options": f"-c statement_timeout={int(statement_timeout_ms)}",
            "connect_timeout": 5,
        }
        self.conn = None

    @classmethod
    def from_config(cls, config: dict, dbname: Optional[str] = None) -> "DatabaseConnection":
        """Create a connection from config.json settings, optionally for a different database."""
        db = dict(config["database"])
        if dbname:
            db["dbname"] = dbname
        return cls(**db, statement_timeout_ms=config["explain"]["statement_timeout_ms"])

    @property
    def dbname(self) -> str:
        """Name of the database this connection uses."""
        return self.params["dbname"]

    def connect(self):
        """Open the read-only connection; raises PlanRetrievalError with a readable message on failure."""
        try:
            self.conn = psycopg2.connect(**self.params)
        except psycopg2.OperationalError as e:
            raise PlanRetrievalError(f"Could not connect to PostgreSQL database '{self.dbname}':\n{e}") from e
        self.conn.set_session(readonly=True, autocommit=True)
        return self

    def close(self):
        """Close the connection if it is open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Connect when used in a `with` block."""
        return self.connect()

    def __exit__(self, *exc):
        """Close the connection at the end of a `with` block."""
        self.close()

    def fetch_all(self, sql: str, params=None) -> list:
        """Run an SQL statement and return all its rows."""
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def list_databases(self) -> list:
        """Names of all databases on the server the user could pick from."""
        rows = self.fetch_all(
            "SELECT datname FROM pg_database WHERE NOT datistemplate AND datallowconn ORDER BY datname"
        )
        return [r[0] for r in rows]

    def get_schema_info(self) -> SchemaInfo:
        """Read tables, columns, keys, indexes and statistics from the system catalogs."""
        tables = {}
        for schema, name, reltuples, relpages in self.fetch_all(f"""
                SELECT n.nspname, c.relname, c.reltuples::bigint, c.relpages
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relkind IN ('r', 'p') AND {_USER_SCHEMAS}
                ORDER BY n.nspname, c.relname"""):
            row_estimate = int(reltuples) if reltuples >= 0 else None
            tables[(schema, name)] = TableInfo(schema, name, row_estimate, relpages)

        for schema, table, col, col_type, not_null in self.fetch_all(f"""
                SELECT n.nspname, c.relname, a.attname,
                       format_type(a.atttypid, a.atttypmod), a.attnotnull
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE a.attnum > 0 AND NOT a.attisdropped
                  AND c.relkind IN ('r', 'p') AND {_USER_SCHEMAS}
                ORDER BY n.nspname, c.relname, a.attnum"""):
            tables[(schema, table)].columns.append((col, col_type, not_null))

        for schema, table, kind, con_name, definition in self.fetch_all(f"""
                SELECT n.nspname, c.relname, con.contype, con.conname,
                       pg_get_constraintdef(con.oid)
                FROM pg_constraint con
                JOIN pg_class c ON c.oid = con.conrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE con.contype IN ('p', 'f', 'u') AND {_USER_SCHEMAS}
                ORDER BY n.nspname, c.relname, con.contype, con.conname"""):
            kind_name = {"p": "Primary key", "f": "Foreign key", "u": "Unique"}[kind]
            tables[(schema, table)].constraints.append((kind_name, con_name, definition))

        for schema, table, idx_name, definition in self.fetch_all("""
                SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename, indexname"""):
            if (schema, table) in tables:
                tables[(schema, table)].indexes.append((idx_name, definition))

        version = self.fetch_all("SHOW server_version")[0][0]
        settings = dict(self.fetch_all(
            "SELECT name, setting || COALESCE(unit, '') FROM pg_settings WHERE name = ANY(%s) ORDER BY name",
            (_PLANNER_SETTINGS,),
        ))
        return SchemaInfo(self.dbname, version, list(tables.values()), settings)

    def get_postgres_plan(self, sql: str) -> QueryPlan:
        """Return PostgreSQL's chosen plan for `sql` without executing the query."""
        query = clean_query(sql)
        try:
            rows = self.fetch_all(f"EXPLAIN (FORMAT JSON, VERBOSE FALSE, COSTS TRUE) {query}")
        except psycopg2.Error as e:
            message = (e.pgerror or str(e)).strip()
            raise PlanRetrievalError(f"PostgreSQL could not plan this query:\n{message}") from e
        raw = rows[0][0]
        return QueryPlan(root=PlanNode.from_explain_json(raw[0]["Plan"]), source="PostgreSQL", raw=raw)


def clean_query(sql: str) -> str:
    """Strip whitespace and trailing semicolons, and reject inputs we can't EXPLAIN."""
    query = sql.strip().rstrip(";").strip()
    if not query:
        raise PlanRetrievalError("Please enter an SQL query.")
    first_word = query.split(None, 1)[0].lower().lstrip("(")
    if first_word == "explain":
        raise PlanRetrievalError("Enter the query itself without EXPLAIN - the tool adds it for you.")
    if first_word not in ("select", "with", "values", "table"):
        raise PlanRetrievalError("Only read queries (SELECT / WITH ...) can be compared.")
    return query


SYSTEM_PROMPT = """You are the query optimizer of a PostgreSQL {version} database.
Given the database schema, table statistics, planner settings and an SQL query,
produce the query execution plan that PostgreSQL's optimizer would choose.

Answer with ONLY a JSON object (no markdown, no commentary) of this form:
{{
  "Plan": <root plan node>,
  "Reasoning": "<2-5 sentences on the key choices: scan methods, join order, join algorithms>"
}}

Each plan node is an object using PostgreSQL EXPLAIN (FORMAT JSON) keys:
  "Node Type"     (required) one of: {node_types}
  "Plans"         list of child nodes (omit or [] for leaf nodes)
  "Relation Name" base table name, for scan nodes
  "Alias"         alias used in the query, for scan nodes
  "Index Name"    index used, for index / bitmap index scans (only indexes listed in the schema exist)
  "Join Type"     Inner, Left, Right, Full, Semi or Anti, for join nodes
  "Strategy"      Plain, Sorted or Hashed, for Aggregate nodes
  "Hash Cond" / "Merge Cond" / "Join Filter" / "Index Cond" / "Filter"   predicates as strings
  "Sort Key" / "Group Key"   lists of expressions
  "Plan Rows"     estimated output rows (number)
  "Total Cost"    estimated total cost in PostgreSQL cost units (number)

Follow PostgreSQL conventions exactly: a Hash Join's inner (second) child is a Hash node;
a Bitmap Heap Scan has a Bitmap Index Scan child; parallel plans use Gather / Gather Merge."""

USER_PROMPT = """Schema and statistics:
{schema}

SQL query:
{query}"""

POSTGRES_NODE_TYPES = [
    "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan",
    "BitmapAnd", "BitmapOr", "Tid Scan", "Subquery Scan", "Function Scan", "Values Scan",
    "CTE Scan", "WorkTable Scan", "Sample Scan", "Nested Loop", "Hash Join", "Merge Join",
    "Hash", "Sort", "Incremental Sort", "Aggregate", "Group", "WindowAgg", "Limit", "Unique",
    "Materialize", "Memoize", "Gather", "Gather Merge", "Append", "Merge Append", "Result",
    "SetOp", "Recursive Union", "ProjectSet", "LockRows",
]


def _name_key(name: str) -> str:
    """'Hash  join' / 'HashJoin' / 'hash_join' -> 'hashjoin'."""
    return "".join(ch for ch in name.lower() if ch.isalpha())


NODE_TYPE_SYNONYMS = {
    **{_name_key(t): (t, {}) for t in POSTGRES_NODE_TYPES},
    "sequentialscan": ("Seq Scan", {}), "tablescan": ("Seq Scan", {}),
    "fullscan": ("Seq Scan", {}), "fulltablescan": ("Seq Scan", {}),
    "indexrangescan": ("Index Scan", {}), "indexseek": ("Index Scan", {}),
    "bitmapscan": ("Bitmap Heap Scan", {}),
    "nestloop": ("Nested Loop", {}), "nestedloopjoin": ("Nested Loop", {}),
    "sortmergejoin": ("Merge Join", {}),
    "hashaggregate": ("Aggregate", {"Strategy": "Hashed"}),
    "groupaggregate": ("Aggregate", {"Strategy": "Sorted"}),
    "sortedaggregate": ("Aggregate", {"Strategy": "Sorted"}),
    "plainaggregate": ("Aggregate", {"Strategy": "Plain"}),
    "topnsort": ("Sort", {}), "topnheapsort": ("Sort", {}),
    "windowaggregate": ("WindowAgg", {}),
}

_JOIN_TYPE_WORDS = [("leftouter", "Left"), ("rightouter", "Right"), ("fullouter", "Full"),
                    ("semi", "Semi"), ("anti", "Anti"), ("left", "Left"), ("right", "Right"),
                    ("full", "Full"), ("inner", "Inner")]

_SCAN_NODES_NEEDING_RELATION = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
                                "Tid Scan", "Sample Scan"}
_MAX_PLAN_DEPTH = 60
_MAX_PLAN_NODES = 500

_JOIN_NODE_TYPES = {"Nested Loop", "Hash Join", "Merge Join"}
_LEAF_NODE_TYPES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan", "Tid Scan", "Sample Scan",
                    "Function Scan", "Values Scan", "CTE Scan", "WorkTable Scan"}
_UNARY_NODE_TYPES = {"Hash", "Sort", "Incremental Sort", "Limit", "Aggregate", "Group", "Unique", "WindowAgg",
                     "Materialize", "Memoize", "Gather", "Gather Merge", "Subquery Scan", "LockRows",
                     "ProjectSet", "SetOp"}
_BITMAP_INDEX_TYPES = {"Bitmap Index Scan", "BitmapAnd", "BitmapOr"}
_SUBPLAN_ROLES = {"InitPlan", "SubPlan"}

_TEXT_FIELDS = ["Relation Name", "Alias", "Index Name", "Join Type", "Strategy", "Partial Mode",
                "CTE Name", "Function Name", "Parent Relationship", "Subplan Name", "Scan Direction"]
_ALLOWED_VALUES = {
    "Join Type": ["Inner", "Left", "Right", "Full", "Semi", "Anti", "Right Semi", "Right Anti"],
    "Strategy": ["Plain", "Sorted", "Hashed", "Mixed"],
    "Partial Mode": ["Simple", "Partial", "Finalize"],
    "Parent Relationship": ["Outer", "Inner", "Member", "Subquery", "InitPlan", "SubPlan"],
}

_SQL_WORDS = {
    "and", "or", "not", "is", "null", "true", "false", "like", "ilike", "in", "any", "all", "some", "between",
    "case", "when", "then", "else", "end", "exists", "array", "as", "from", "for", "to", "at", "on", "using",
    "date", "interval", "timestamp", "time", "zone", "with", "without", "year", "month", "day", "hour",
    "minute", "second", "numeric", "integer", "int", "bigint", "smallint", "text", "bpchar", "varchar",
    "character", "varying", "double", "precision", "real", "boolean", "bool", "asc", "desc", "nulls",
    "first", "last", "distinct", "subplan", "initplan", "hashed", "returns", "row", "escape", "similar",
}
_CAST_TYPES = re.compile(r"::\s*(timestamp|time) with(out)? time zone|::\s*character varying|"
                         r"::\s*double precision|::\s*\w+(\(\d+(,\s*\d+)?\))?(\[\])?", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER = re.compile(r"(?<![\w$.])([A-Za-z_]\w*)(?:\s*\.\s*([A-Za-z_]\w*))?(?![\w(])(?!\s*\()")


class InvalidAIResponse(Exception):
    """The AI's answer can't be turned into a plan; the message is fed back to the AI."""


@dataclass
class _Validation:
    """Collects problems found while checking one AI answer against the schema and the query.

    Attributes:
        query_words: identifiers in the user's query - aliases and output names, which are
                     not hallucinations even though they aren't columns
        warnings:    every problem found
        serious:     the problems that suggest hallucination and justify asking the AI again
        node_count:  nodes seen so far, to reject absurdly large plans
    """

    schema: "SchemaInfo"
    query_words: set = field(default_factory=set)
    warnings: list = field(default_factory=list)
    serious: list = field(default_factory=list)
    node_count: int = 0

    def __post_init__(self):
        """Precompute the table names, column names and the table each index belongs to."""
        self.tables = {t.name.lower(): t for t in self.schema.tables}
        self.columns = {c[0].lower() for t in self.schema.tables for c in t.columns}
        self.index_owner = {i[0]: t.name.lower() for t in self.schema.tables for i in t.indexes}

    def warn(self, message: str, serious: bool = False):
        """Record a warning once; serious ones are also added to `serious`."""
        if message not in self.warnings:
            self.warnings.append(message)
        if serious and message not in self.serious:
            self.serious.append(message)


def build_prompt(sql: str, schema: SchemaInfo) -> tuple:
    """Return (system prompt, user prompt) for the plan request."""
    system = SYSTEM_PROMPT.format(
        version=schema.server_version.split()[0],
        node_types=", ".join(POSTGRES_NODE_TYPES),
    )
    return system, USER_PROMPT.format(schema=schema.to_prompt_text(), query=sql)


def extract_json(text: str) -> object:
    """Pull the first JSON object/array out of a model answer."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch in "{[":
                try:
                    return decoder.raw_decode(text[i:])[0]
                except json.JSONDecodeError:
                    continue
    except RecursionError:
        raise InvalidAIResponse("The plan is nested unreasonably deep.") from None
    raise InvalidAIResponse("The response did not contain valid JSON.")


def _to_number(value) -> Optional[float]:
    """Parse a number such as 1234, "1,234" or "1.5e3"; None if it isn't a usable number."""
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check_text_fields(clean: dict, raw_type: str, v: _Validation):
    """Fields that must be text; drop anything else."""
    for key in _TEXT_FIELDS:
        if key not in clean:
            continue
        value = clean[key]
        if not isinstance(value, str) or not value.strip():
            v.warn(f'{raw_type}: "{key}" should be text but is {json.dumps(value)[:40]}; ignored.')
            del clean[key]
        elif key in _ALLOWED_VALUES:
            allowed = {a.lower(): a for a in _ALLOWED_VALUES[key]}
            if value.strip().lower() in allowed:
                clean[key] = allowed[value.strip().lower()]
            else:
                v.warn(f'{raw_type}: "{key}" has an invalid value "{value}"; ignored.')
                del clean[key]

    for key in CONDITION_KEYS:
        if key in clean:
            value = clean[key]
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                clean[key] = " AND ".join(value)
            elif not isinstance(value, str):
                v.warn(f'{raw_type}: "{key}" should be text; ignored.')
                del clean[key]
    for key in ("Sort Key", "Group Key"):
        if key in clean:
            value = clean[key]
            if isinstance(value, str):
                clean[key] = [value]
            elif not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                v.warn(f'{raw_type}: "{key}" should be a list of expressions; ignored.')
                del clean[key]
    if "Parallel Aware" in clean and not isinstance(clean["Parallel Aware"], bool):
        del clean["Parallel Aware"]


def _check_numbers(clean: dict, node_type: str, v: _Validation):
    """Rows and costs must be finite, non-negative numbers; anything else is dropped with a warning."""
    for key in ("Plan Rows", "Total Cost", "Startup Cost"):
        if key in clean:
            number = _to_number(clean[key])
            if number is None:
                v.warn(f'{node_type}: "{key}" is not a number ({json.dumps(clean[key])[:30]}); ignored.')
            elif number < 0:
                v.warn(f'{node_type}: "{key}" is negative ({number:g}); ignored.')
                number = None
            clean[key] = number


def _check_schema_objects(clean: dict, node_type: str, v: _Validation):
    """Tables and indexes must exist, and an index must belong to the table being scanned."""
    relation = clean.get("Relation Name")
    if relation is not None:
        relation = relation.split(".")[-1].strip().strip('"').lower()
        clean["Relation Name"] = relation
        if (relation not in v.tables and node_type in ("Subquery Scan", "CTE Scan")
                and relation in v.query_words):
            del clean["Relation Name"]
            clean.setdefault("Alias", relation)
            if node_type == "CTE Scan":
                clean.setdefault("CTE Name", relation)
            v.warn(f'{node_type} gives its alias "{relation}" as a table name; treated as the alias.')
            relation = None
        elif relation not in v.tables:
            v.warn(f'{node_type} reads table "{relation}", which does not exist.', serious=True)
    elif node_type in _SCAN_NODES_NEEDING_RELATION:
        v.warn(f"{node_type} does not say which table it reads.", serious=True)

    index = clean.get("Index Name")
    if index is not None:
        index = index.split(".")[-1].strip().strip('"')
        clean["Index Name"] = index
        owner = v.index_owner.get(index)
        if owner is None:
            v.warn(f'{node_type} uses index "{index}", which does not exist.', serious=True)
        elif relation in v.tables and owner != relation:
            v.warn(f'Index "{index}" belongs to {owner}, not {relation}.', serious=True)
    elif node_type in ("Index Scan", "Index Only Scan", "Bitmap Index Scan"):
        v.warn(f"{node_type} does not say which index it uses.")


def _check_columns(clean: dict, node_type: str, v: _Validation):
    """Columns named in conditions must exist in the schema (or at least appear in the query)."""
    for key in CONDITION_KEYS:
        if not isinstance(clean.get(key), str):
            continue
        text = _CAST_TYPES.sub(" ", _STRING_LITERAL.sub(" ", clean[key]))
        for match in _IDENTIFIER.finditer(text):
            column = (match.group(2) or match.group(1)).lower()
            if column in _SQL_WORDS or column in v.columns or column in v.query_words:
                continue
            v.warn(f'{node_type} {key} refers to column "{column}", which does not exist.', serious=True)


def _check_structure(clean: dict, node_type: str, v: _Validation):
    """Each kind of node needs a particular number/kind of inputs, e.g. a join needs exactly two."""
    inputs = [c for c in clean["Plans"] if c.get("Parent Relationship") not in _SUBPLAN_ROLES]
    count = len(inputs)
    where = f" on {clean['Relation Name']}" if clean.get("Relation Name") else ""
    if node_type in _JOIN_NODE_TYPES:
        if count != 2:
            v.warn(f"{node_type} has {count} input(s); a join needs exactly 2.", serious=True)
        elif node_type == "Hash Join" and inputs[1]["Node Type"] != "Hash":
            v.warn("Hash Join's second input should be a Hash node (the hash table it builds).")
    elif node_type == "Bitmap Heap Scan":
        if not any(c["Node Type"] in _BITMAP_INDEX_TYPES for c in inputs):
            v.warn(f"Bitmap Heap Scan{where} has no Bitmap Index Scan beneath it.")
    elif node_type in ("BitmapAnd", "BitmapOr"):
        if count < 2:
            v.warn(f"{node_type} combines {count} bitmap(s); it needs at least 2.")
    elif node_type in _LEAF_NODE_TYPES:
        if count:
            v.warn(f"{node_type}{where} reads directly from storage but has {count} input step(s).")
    elif node_type in _UNARY_NODE_TYPES and count != 1:
        v.warn(f"{node_type} should have exactly 1 input but has {count}.")


def normalise_ai_node(node: object, v: _Validation, depth: int = 0) -> dict:
    """Check one AI plan node (recursively) and rewrite it into PostgreSQL EXPLAIN JSON form."""
    if depth > _MAX_PLAN_DEPTH:
        raise InvalidAIResponse("The plan is nested unreasonably deep.")
    v.node_count += 1
    if v.node_count > _MAX_PLAN_NODES:
        raise InvalidAIResponse(f"The plan has more than {_MAX_PLAN_NODES} steps.")
    if not isinstance(node, dict):
        raise InvalidAIResponse(f"Expected a plan node object, got {type(node).__name__}.")

    raw_type = next((node[k] for k in ("Node Type", "NodeType", "node_type", "type") if k in node), None)
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise InvalidAIResponse('A plan node is missing its "Node Type".')

    clean = {k: value for k, value in node.items() if k not in ("Plans", "children", "NodeType", "node_type", "type")}
    _check_text_fields(clean, raw_type.strip(), v)

    key = _name_key(raw_type)
    if key.startswith("parallel"):
        key, clean["Parallel Aware"] = key[len("parallel"):], True
    for mode in ("partial", "finalize"):
        if key.startswith(mode):
            key, clean["Partial Mode"] = key[len(mode):], mode.capitalize()
    if key not in NODE_TYPE_SYNONYMS and ("join" in key or "loop" in key):
        for word, join_type in _JOIN_TYPE_WORDS:
            stripped = key.replace(word, "", 1).replace("outer", "", 1)
            if word in key and stripped in NODE_TYPE_SYNONYMS:
                key = stripped
                clean.setdefault("Join Type", join_type)
                break
    if key in NODE_TYPE_SYNONYMS:
        node_type, extra = NODE_TYPE_SYNONYMS[key]
        for k, value in extra.items():
            clean.setdefault(k, value)
    else:
        node_type = raw_type.strip()
        v.warn(f'Unrecognised operator "{node_type}" (not a PostgreSQL node type).')
    clean["Node Type"] = node_type

    _check_schema_objects(clean, node_type, v)
    _check_columns(clean, node_type, v)
    _check_numbers(clean, node_type, v)

    children = node.get("Plans", node.get("children", []))
    if children is None:
        children = []
    if not isinstance(children, list):
        raise InvalidAIResponse('"Plans" must be a list of child nodes.')
    clean["Plans"] = [normalise_ai_node(child, v, depth + 1) for child in children]
    _check_structure(clean, node_type, v)
    return clean


def parse_ai_response(text: str, schema: SchemaInfo, expected_tables=None, query: str = "") -> tuple:
    """Turn the raw model answer into (PlanNode root, reasoning, warnings, serious warnings, parsed JSON)."""
    data = extract_json(text)
    if isinstance(data, list):
        if not data:
            raise InvalidAIResponse("The response was an empty list.")
        data = data[0]
    if not isinstance(data, dict):
        raise InvalidAIResponse("The response JSON is not an object.")

    plan = data.get("Plan", data if "Node Type" in data else None)
    if plan is None:
        raise InvalidAIResponse('The response has no "Plan" object.')
    reasoning = data.get("Reasoning", "")

    v = _Validation(schema, query_words={w.lower() for w in re.findall(r"[A-Za-z_]\w*", query)})
    root = PlanNode.from_explain_json(normalise_ai_node(plan, v))

    if expected_tables:
        expected = {t.lower() for t in expected_tables}
        used = {n.relation for n in root.walk() if n.relation}
        if not used & expected:
            raise InvalidAIResponse(f"The plan reads none of the tables the query uses "
                                    f"({', '.join(sorted(expected))}).")
        for table in sorted(expected - used):
            v.warn(f'The plan never reads table "{table}", which the query uses.', serious=True)
        for table in sorted((used - expected) & set(v.tables)):
            v.warn(f'The plan reads table "{table}", which the query does not use.', serious=True)

    reasoning = reasoning if isinstance(reasoning, str) else json.dumps(reasoning)
    return root, reasoning, v.warnings, v.serious, data


DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4.1",
    "claude": "claude-opus-5-5",
}


def _require(module_name: str, package: str):
    """Import an AI provider's SDK only when that provider is used, with a clear message if it is missing."""
    try:
        return __import__(module_name, fromlist=["_"])
    except ImportError as e:
        raise PlanRetrievalError(f"This AI provider needs the '{package}' package:  pip install {package}") from e


def _call_gemini(model: str, system: str, messages: list, api_key: str, ai_config: dict) -> str:
    """Ask a Gemini model and return its text answer."""
    genai = _require("google.genai", "google-genai")
    types = _require("google.genai.types", "google-genai")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=int(ai_config.get("timeout_s", 180)) * 1000,
        retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0,
                                             http_status_codes=[429, 500, 502, 503, 504]),
    ))
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    extra = {}
    if ai_config.get("temperature") is not None:
        extra["temperature"] = ai_config["temperature"]
    if ai_config.get("effort"):
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=ai_config["effort"].upper())
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            **extra,
        ),
    )
    return response.text or ""


def _call_openai(model: str, system: str, messages: list, api_key: str, ai_config: dict) -> str:
    """Ask an OpenAI model in JSON mode and return its text answer."""
    openai = _require("openai", "openai")
    client = openai.OpenAI(api_key=api_key)
    extra = {}
    if ai_config.get("temperature") is not None:
        extra["temperature"] = ai_config["temperature"]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        response_format={"type": "json_object"},
        **extra,
    )
    return response.choices[0].message.content or ""


def _call_claude(model: str, system: str, messages: list, api_key: str, ai_config: dict) -> str:
    """Ask a Claude model and return its text answer."""
    anthropic = _require("anthropic", "anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=messages,
        output_config={"effort": ai_config.get("effort", "high")},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise PlanRetrievalError("Claude declined to answer this request.")
    return "".join(block.text for block in response.content if block.type == "text")


PROVIDERS = {"gemini": _call_gemini, "openai": _call_openai, "claude": _call_claude}


def get_ai_plan(sql: str, schema: SchemaInfo, config: dict, expected_tables=None) -> QueryPlan:
    """Ask the configured AI agent for a plan and parse it into a PlanNode tree.

    Builds the prompt from the schema and the query, calls the provider and validates
    the answer. If the answer is unusable, or has serious warnings (made-up tables,
    indexes or columns, missing tables), the model is told what was wrong and asked
    again, up to max_retries extra calls.

    expected_tables: base tables the query uses (e.g. postgres_plan.root.base_relations()),
    used to catch plans that leave out or invent tables.
    """
    ai_config = config["ai"]
    provider = ai_config.get("provider", "gemini").lower()
    if provider not in PROVIDERS:
        raise PlanRetrievalError(f"Unknown AI provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    model = ai_config.get("model") or DEFAULT_MODELS[provider]
    api_key = get_api_key(provider)
    if not api_key:
        raise PlanRetrievalError(
            f"No API key for '{provider}'. Add {provider.upper()}_API_KEY=... to the .env file "
            "in the project folder (see .env.example)."
        )

    query = clean_query(sql)
    system, user = build_prompt(query, schema)

    messages = [{"role": "user", "content": user}]
    attempts = 1 + int(ai_config.get("max_retries", 2))
    warning_retries = int(ai_config.get("warning_retries", 1))
    best, last_error = None, None
    for attempt in range(attempts):
        try:
            answer = PROVIDERS[provider](model, system, messages, api_key, ai_config)
        except PlanRetrievalError:
            raise
        except Exception as e:
            raise PlanRetrievalError(f"The {provider} API request failed:\n{e}") from e
        try:
            root, reasoning, warnings, serious, data = parse_ai_response(answer, schema, expected_tables, query)
        except InvalidAIResponse as e:
            last_error = e
            feedback = f"That answer could not be used: {e} Reply again with only the JSON object described."
        else:
            plan = QueryPlan(root=root, source=model, raw=data, reasoning=reasoning,
                             warnings=warnings, serious_warnings=serious)
            if best is None or len(plan.serious_warnings) < len(best.serious_warnings):
                best = plan
            if not plan.serious_warnings or warning_retries <= 0 or attempt == attempts - 1:
                break
            warning_retries -= 1
            problems = "\n".join(f"- {w}" for w in plan.serious_warnings)
            feedback = (f"Your plan has these problems:\n{problems}\nUse only the tables, columns and "
                        "indexes listed in the schema, and read every table the query uses. "
                        "Reply again with only the corrected JSON object.")
        messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": feedback}]

    if best is None:
        raise PlanRetrievalError(f"The AI did not return a usable plan after several attempts:\n{last_error}")
    best.attempts = attempt + 1
    return best


if __name__ == "__main__":
    test_query = """
        SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount)
                                 ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= date '1993-11-01'
          AND l_shipdate < date '1993-11-01' + interval '1' month
    """
    cfg = load_config()
    with DatabaseConnection.from_config(cfg) as db:
        print("Databases on server:", ", ".join(db.list_databases()))
        print("\n=== Schema (as sent to the AI) ===")
        print(db.get_schema_info().to_prompt_text())
        print("\n=== PostgreSQL plan ===")
        print(db.get_postgres_plan(test_query).root.to_text())
