# SC3020 Project 2 – QEP Comparator: PostgreSQL vs an AI agent

## 1. Install PostgreSQL 16

### Windows
1. Download the installer from https://www.postgresql.org/download/windows/ (EnterpriseDB, version 16).
2. Run it with the default options. When asked, **set a password for the `postgres` user** and write it
   down. You will need it in steps 2 and 4.
3. The PostgreSQL service starts automatically, now and on every boot.
4. Add PostgreSQL's tools to PATH so `psql` and `createdb` work in Command Prompt:
   `C:\Program Files\PostgreSQL\16\bin`
   (Start → "Edit the system environment variables" → Environment Variables → select **Path** under
   *User variables* → Edit → New.)
   Open a **new** Command Prompt afterwards.

**Login on Windows:** user `postgres`, with the password from step 2.

### macOS (Homebrew)
```bash
brew install postgresql@16
brew services start postgresql@16
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```
**Login on macOS:** your Mac username, no password (Homebrew trusts local connections).

---

## 2. Create the TPC-H database

If the database is already loaded in PostgreSQL, skip to step 3 and set `dbname` in step 4a.

### 2a. Generate the data files
Download **TPC-H Tools** from https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp
and unzip it. The generator is in the `dbgen` folder.

- **Windows:** install Visual Studio Community (free) with the **Desktop development with C++** workload.
  Open `dbgen\tpch.sln` and click **OK** on the *One-way upgrade* prompt. Keep the **Debug | Win32**
  configuration (Release is missing settings dbgen needs). In Solution Explorer (View → Solution Explorer),
  right-click **dbgen** → **Build**.
  Don't use *Build Solution*: it also builds `qgen`, which fails and isn't needed. Then, in Command Prompt:
  ```
  cd C:\path\to\dbgen
  Debug\dbgen.exe -s 1
  ```
  dbgen must be run from inside the `dbgen` folder (it reads `dists.dss` there); the `.tbl` files appear there too.
- **macOS:** in the `dbgen` folder, copy `makefile.suite` to `makefile` and set
  `CC=gcc`, `DATABASE=ORACLE`, `MACHINE=LINUX`, `WORKLOAD=TPCH`. Run `make`, then `./dbgen -s 1`.
  If the build complains about `malloc.h`, change `#include <malloc.h>` to `#include <stdlib.h>` in the
  files it names.

You should now have 8 files: `customer.tbl`, `lineitem.tbl`, `nation.tbl`, `orders.tbl`, `part.tbl`,
`partsupp.tbl`, `region.tbl`, `supplier.tbl` (about 1 GB in total at scale factor 1).

### 2b. Load the data
Create an empty database:
```bash
createdb -U postgres tpch        # Windows
createdb tpch                    # macOS
```
Load the files. This step needs Python and the packages from **step 3**, so do step 3 first if you haven't.
Run it from the project folder with the virtual environment activated:
```bash
python tools\load_tpch.py --data-dir C:\path\to\dbgen --dbname tpch --user postgres --password YOUR_PASSWORD   # Windows
python tools/load_tpch.py --data-dir "path/to/dbgen" --dbname tpch                                             # macOS
```
`--data-dir` is the folder holding the 8 `.tbl` files (the `dbgen` folder, unless you moved them).

### 2c. Check it worked
```bash
psql -U postgres -d tpch -c "SELECT COUNT(*) FROM lineitem;"     # Windows
psql -d tpch -c "SELECT COUNT(*) FROM lineitem;"                 # macOS
```
Expected result: **6001215**.

---

## 3. Set up Python

Install **Python 3.11**:
- **Windows:** https://www.python.org/downloads/ – tick **"Add python.exe to PATH"** in the installer.
  Tkinter (needed for the window) is included.
- **macOS:** `brew install python@3.11 python-tk@3.11` (Homebrew's Python needs `python-tk` for the window).

Then, in a terminal **in the project folder**:
```bash
# Windows (Command Prompt)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Configure the project

### 4a. Database login – `config.json`
Edit the `database` section to match your machine:

| Field | Windows | macOS |
|---|---|---|
| `user` | `"postgres"` (already set) | `""` (delete the default `postgres`) |
| `password` | your installer password | `""` (leave empty) |
| `host`, `port` | `"localhost"`, `5432` | `"localhost"`, `5432` |
| `dbname` | the database to open first, e.g. `"tpch"` | same |

### 4b. AI model – `config.json`
The `ai` section chooses the AI agent. The default is Claude Haiku 5.5:
```json
"ai": {
  "provider": "claude",
  "model": "claude-haiku-5-5",
  "max_retries": 2,
  "temperature": null,
  "effort": "low"
}
```

### 4c. AI API key – `.env`
The submitted zip already includes a `.env` file with a Claude API key for marking (spending capped at
US$3). The key is meant for the default model, Claude Haiku 5.5, so leave the AI provider and model in the
program's top bar as they are, and skip the rest of this step.

To use your own key or another provider, put its key in `.env` in the project folder (copy `.env.example`
to `.env` first if there is none), one key per line:
```
CLAUDE_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENAI_API_KEY=sk-...
```
Then choose that provider in the program's top bar.

---

## 5. Run

From the project folder:
```bash
# Windows (Command Prompt)
.venv\Scripts\activate
python project.py

# macOS
source .venv/bin/activate
python project.py
```
In PyCharm: set the project interpreter to `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python`
(macOS), then run `project.py`.

### Using the program
1. Check the **top bar**: the database (e.g. `tpch`) and the AI provider/model.
2. Type or paste an SQL query in the **SQL query** box (only read queries – `SELECT` / `WITH` – are accepted;
   the database is opened read-only and the query is never executed, only planned).
3. Click **Compare plans**. PostgreSQL's plan appears almost at once; the AI's plan follows after roughly
   5–45 seconds, depending on the model.
4. Read the results:
   - **Tree view** – both plans as trees. Each box is one step; arrows point the way rows flow.
     Colours: **green** = same in both plans, **amber** = same step but a different method,
     **red** = only in this plan, **grey** = not compared. Use **−**, **+** and **Fit** to zoom, drag to pan,
     and the mouse wheel to scroll (Shift + wheel for sideways).
   - **Click any step** to highlight its counterpart in the other plan and see the details and the
     explanation in the **Selected step** panel.
   - **Comparison** panel – the **difference rate** (0% = identical plans, 100% = nothing in common), every
     difference with its explanation, the similarities, and notes that don't count towards the score.
   - **Text view** – both plans as indented text, plus the AI's own reasoning and any **validation
     warnings** (made-up tables/indexes/columns, missing tables, impossible plan structure).

---

