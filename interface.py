"""interface.py - Graphical user interface (Tkinter).

Layout:
  - Top bar:     database selector, AI provider / model selector
  - Left panel:  SQL query editor and "Compare plans" button
  - Right panel: PostgreSQL plan and AI plan side by side, colour-coded
                 (green = same, amber = changed, red = only in this plan),
                 as drawn trees ("Tree view" tab) or indented text ("Text view")
  - Bottom:      difference rate, differences, similarities and notes, plus
                 details of the step clicked in a tree
  - Status bar

Clicking a step in one tree highlights its counterpart in the other tree and
shows why the two differ. STATUS_COLOURS holds each status's light background
colour and legend text.
"""

import copy
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

from comparison import MatchStatus, compare_plans
from preprocessing import DEFAULT_MODELS, PROVIDERS, DatabaseConnection, PlanRetrievalError, get_ai_plan

SAMPLE_QUERY = """SELECT c_name, sum(o_totalprice) AS total_spent
FROM customer, orders
WHERE c_custkey = o_custkey
  AND o_orderdate >= date '1995-01-01'
GROUP BY c_name
ORDER BY total_spent DESC
LIMIT 10;"""

STATUS_COLOURS = {
    MatchStatus.SAME: ("#dcfce7", "Same in both plans"),
    MatchStatus.CHANGED: ("#fef3c7", "Same step, different method"),
    MatchStatus.ONLY_IN_ONE: ("#fee2e2", "Only in this plan"),
}


def enable_high_dpi():
    """Stop Tkinter looking blurry on high-resolution Windows screens."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


class PlanCanvas(ttk.LabelFrame):
    """Draws one plan as a top-down tree of boxes, coloured by comparison status..

    Attributes:
        node_by_tag: canvas tag -> PlanNode, to find the node under the mouse
        rect_of:     id(PlanNode) -> its rectangle item, to highlight it
        note:        extra toolbar text, e.g. the AI plan's warning counts
    """

    BORDERS = {MatchStatus.SAME: "#16a34a", MatchStatus.CHANGED: "#d97706",
               MatchStatus.ONLY_IN_ONE: "#dc2626", None: "#9ca3af"}
    UNCOMPARED_FILL = "#f3f4f6"
    SELECTED_BORDER = "#2563eb"
    MARGIN = 16

    def __init__(self, parent, title: str, on_select):
        """Build the toolbar (zoom buttons, info label) and the scrollable canvas, and bind the mouse."""
        super().__init__(parent, text=title, padding=4)
        self.on_select = on_select
        self.plan = None
        self.node_status = {}
        self.zoom = 1.0
        self.message = "No plan yet - enter a query and click Compare plans."
        self.node_by_tag = {}
        self.rect_of = {}
        self.selected = None
        self.note = ""

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        for text, command in (("−", lambda: self.set_zoom(self.zoom / 1.25)),
                              ("+", lambda: self.set_zoom(self.zoom * 1.25)),
                              ("Fit", self.fit)):
            ttk.Button(toolbar, text=text, width=4, command=command).pack(side=tk.LEFT, padx=(0, 4))
        self.info_label = ttk.Label(toolbar, text="")
        self.info_label.pack(side=tk.RIGHT)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, background="white", highlightthickness=0)
        y_scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", lambda e: self._on_wheel(e, horizontal=False))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self._on_wheel(e, horizontal=True))
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.canvas.bind("<Configure>", lambda e: self._draw() if self.plan is None else None)


    def show(self, plan, node_status=None, fit=True, note=""):
        """Draw a plan (optionally coloured by comparison status)."""
        self.plan, self.node_status, self.selected, self.note = plan, node_status or {}, None, note
        self._draw()
        if fit:
            self.after_idle(self.fit)

    def show_message(self, message: str):
        """Clear the drawing and show a message instead (e.g. while waiting, or on error)."""
        self.plan, self.node_status, self.selected, self.message = None, {}, None, message
        self.info_label.configure(text="")
        self._draw()

    def set_zoom(self, zoom: float):
        """Redraw at a new zoom level (30% to 200%), keeping the current selection."""
        self.zoom = min(2.0, max(0.3, zoom))
        selected = self.selected
        self._draw()
        if selected is not None:
            self.select(selected)

    def fit(self):
        """Zoom so the whole tree fits the visible width."""
        if self.plan is None:
            return
        bbox = self.canvas.bbox("all")
        visible = self.canvas.winfo_width()
        if not bbox or visible <= 1:
            return
        tree_width = (bbox[2] - bbox[0]) / self.zoom
        self.set_zoom(max(0.55, min(1.0, (visible - 2 * self.MARGIN) / max(tree_width, 1))))

    def select(self, node, scroll=True):
        """Highlight one node's box (and un-highlight the previous one)."""
        self.clear_selection()
        rect = self.rect_of.get(id(node))
        if rect is None:
            return
        self.selected = node
        self.canvas.itemconfigure(rect, outline=self.SELECTED_BORDER, width=max(2, round(3 * self.zoom)))
        if scroll:
            self._scroll_into_view(rect)

    def clear_selection(self):
        """Remove the selection outline, restoring the box's normal border."""
        if self.selected is not None and id(self.selected) in self.rect_of:
            status = self.node_status.get(id(self.selected))
            self.canvas.itemconfigure(self.rect_of[id(self.selected)], outline=self.BORDERS[status],
                                      width=max(1, round(1.5 * self.zoom)))
        self.selected = None


    def _fonts(self) -> dict:
        """Bold, normal and small fonts at the current zoom, in the platform's default font."""
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        size = max(6, round(10 * self.zoom))
        return {"bold": tkfont.Font(family=family, size=size, weight="bold"),
                "normal": tkfont.Font(family=family, size=size),
                "small": tkfont.Font(family=family, size=max(6, round(9 * self.zoom)))}

    @staticmethod
    def _node_lines(node) -> list:
        """(text, font) lines shown inside a node's box."""
        title = node.label().split(" using ")[0].split(" on ")[0]
        lines = [(title, "bold")]
        if node.relation:
            alias = f" {node.alias}" if node.alias and node.alias != node.relation else ""
            lines.append((node.relation + alias, "normal"))
        if node.index_name:
            lines.append((f"index: {node.index_name}", "normal"))
        if node.conditions:
            key, value = next(iter(node.conditions.items()))
            lines.append((f"{key}: {value}", "small"))
        estimates = []
        if node.estimated_rows is not None:
            estimates.append(f"rows {node.estimated_rows:,.0f}")
        if node.estimated_cost is not None:
            estimates.append(f"cost {node.estimated_cost:,.0f}")
        if estimates:
            lines.append(("  ·  ".join(estimates), "small"))
        return lines

    @staticmethod
    def _shorten(text: str, font, max_px: float) -> str:
        """Cut text with '…' so it fits in max_px."""
        if font.measure(text) <= max_px:
            return text
        while text and font.measure(text + "…") > max_px:
            text = text[:-1]
        return text + "…"

    def _draw(self):
        """Redraw the whole tree, or the message when there is no plan.

        1. Measure every box from its lines of text, and the tallest box at each depth.
        2. Work out the width of every subtree, so children sit side by side.
        3. Place each parent centred above its children, one row per depth.
        4. Draw the edges first (arrows point up, the way rows flow from a step's inputs
           to the step), then the boxes on top.
        """
        self.canvas.delete("all")
        self.node_by_tag.clear()
        self.rect_of.clear()
        if self.plan is None:
            width = max(self.canvas.winfo_width(), 200)
            self.canvas.create_text(width / 2, 40, text=self.message, fill="#6b7280", width=width - 40,
                                    justify=tk.CENTER)
            self.canvas.configure(scrollregion=(0, 0, width, 100))
            return

        fonts, z = self._fonts(), self.zoom
        pad, h_gap, v_gap = 6 * z, 18 * z, 34 * z
        min_w, max_w = 110 * z, 240 * z

        boxes = {}
        level_height = {}
        def measure(node, depth):
            """Size each box from its lines and record the tallest box at each depth."""
            lines = [(self._shorten(text, fonts[f], max_w - 2 * pad), f) for text, f in self._node_lines(node)]
            width = min(max_w, max(min_w, max(fonts[f].measure(t) for t, f in lines) + 2 * pad))
            height = sum(fonts[f].metrics("linespace") for _, f in lines) + 2 * pad
            boxes[id(node)] = (width, height, lines)
            level_height[depth] = max(level_height.get(depth, 0), height)
            for child in node.children:
                measure(child, depth + 1)
        measure(self.plan.root, 0)

        level_y, y = {}, self.MARGIN
        for depth in sorted(level_height):
            level_y[depth] = y
            y += level_height[depth] + v_gap

        subtree_width = {}
        def span(node):
            """Width of a subtree: the node's own box or its children side by side, whichever is wider."""
            children = sum(span(c) for c in node.children) + h_gap * max(0, len(node.children) - 1)
            subtree_width[id(node)] = max(boxes[id(node)][0], children)
            return subtree_width[id(node)]
        span(self.plan.root)

        position = {}
        def place(node, left, depth):
            """Position a node centred above its children, then place the children left to right."""
            width, height, _ = boxes[id(node)]
            centre = left + subtree_width[id(node)] / 2
            position[id(node)] = (centre - width / 2, level_y[depth], centre + width / 2, level_y[depth] + height)
            children_width = sum(subtree_width[id(c)] for c in node.children) + h_gap * max(0, len(node.children) - 1)
            x = centre - children_width / 2
            for child in node.children:
                place(child, x, depth + 1)
                x += subtree_width[id(child)] + h_gap
        place(self.plan.root, self.MARGIN, 0)

        for node in self.plan.root.walk():
            px0, _, px1, py1 = position[id(node)]
            for child in node.children:
                cx0, cy0, cx1, _ = position[id(child)]
                self.canvas.create_line((cx0 + cx1) / 2, cy0, (px0 + px1) / 2, py1, fill="#9ca3af",
                                        width=max(1, z), arrow=tk.LAST, arrowshape=(8 * z, 10 * z, 3 * z))
        for node in self.plan.root.walk():
            x0, y0, x1, y1 = position[id(node)]
            tag = f"n{len(self.node_by_tag)}"
            self.node_by_tag[tag] = node
            status = self.node_status.get(id(node))
            fill = STATUS_COLOURS[status][0] if status else self.UNCOMPARED_FILL
            self.rect_of[id(node)] = self.canvas.create_rectangle(
                x0, y0, x1, y1, fill=fill, outline=self.BORDERS[status], width=max(1, round(1.5 * z)),
                tags=("node", tag))
            ty = y0 + pad
            for text, f in boxes[id(node)][2]:
                self.canvas.create_text((x0 + x1) / 2, ty, text=text, font=fonts[f], anchor="n",
                                        fill="#111827", tags=("node", tag))
                ty += fonts[f].metrics("linespace")

        bbox = self.canvas.bbox("all")
        self.canvas.configure(scrollregion=(0, 0, bbox[2] + self.MARGIN, bbox[3] + self.MARGIN))
        info = f"{self.plan.root.size()} steps  ·  {round(self.zoom * 100)}%"
        self.info_label.configure(text=f"{self.note}  ·  {info}" if self.note else info)


    def _on_press(self, event):
        """Remember the press point (for dragging) and select the box under the mouse, if any."""
        self.canvas.scan_mark(event.x, event.y)
        for item in self.canvas.find_withtag("current"):
            for tag in self.canvas.gettags(item):
                node = self.node_by_tag.get(tag)
                if node is not None:
                    self.select(node, scroll=False)
                    self.on_select(node, self)
                    return

    def _on_wheel(self, event, horizontal: bool):
        """Scroll one step per wheel event.

        Windows reports multiples of 120 and macOS small values, so only the sign is used.
        """
        step = -1 if event.delta > 0 else 1
        (self.canvas.xview_scroll if horizontal else self.canvas.yview_scroll)(step, "units")

    def _scroll_into_view(self, item):
        """Scroll so the given canvas item is centred in view."""
        x0, y0, x1, y1 = self.canvas.bbox(item)
        region = [float(v) for v in str(self.canvas.cget("scrollregion")).split()]
        total_w, total_h = max(region[2] - region[0], 1), max(region[3] - region[1], 1)
        view_w, view_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.xview_moveto(max(0.0, ((x0 + x1) / 2 - view_w / 2) / total_w))
        self.canvas.yview_moveto(max(0.0, ((y0 + y1) / 2 - view_h / 2) / total_h))


class PlanComparisonApp:
    """Main application window."""

    def __init__(self, root: tk.Tk, config: dict):
        """Read the defaults from the config and build the window."""
        self.root = root
        self.config = config
        self.root.title("QEP Comparator - PostgreSQL vs AI")
        self.root.geometry("1350x850")
        self.root.minsize(950, 600)

        ai = config["ai"]
        provider = ai.get("provider", "gemini")
        self.database_var = tk.StringVar(value=config["database"]["dbname"])
        self.provider_var = tk.StringVar(value=provider)
        self.model_var = tk.StringVar(value=ai.get("model") or DEFAULT_MODELS.get(provider, ""))
        self.status_var = tk.StringVar(value="Ready")
        self.result_queue = queue.Queue()
        self.postgres_plan = self.ai_plan = None

        self._build_top_bar()
        self._build_main_area()
        self._build_status_bar()
        self.refresh_databases()


    def _build_top_bar(self):
        """Database selector, and the AI provider and model selectors."""
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Database:").pack(side=tk.LEFT)
        self.database_box = ttk.Combobox(bar, textvariable=self.database_var, width=18, state="readonly")
        self.database_box.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(bar, text="Refresh", command=self.refresh_databases).pack(side=tk.LEFT)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Label(bar, text="AI provider:").pack(side=tk.LEFT)
        provider_box = ttk.Combobox(bar, textvariable=self.provider_var, values=list(PROVIDERS),
                                    width=8, state="readonly")
        provider_box.pack(side=tk.LEFT, padx=(5, 10))
        provider_box.bind("<<ComboboxSelected>>", self.on_provider_changed)
        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.model_var, width=22).pack(side=tk.LEFT, padx=(5, 10))

    def _build_main_area(self):
        """Query panel and plans on top; comparison report and selected step below. """
        vertical = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical.pack(fill=tk.BOTH, expand=True, padx=10)
        panes = ttk.PanedWindow(vertical, orient=tk.HORIZONTAL)
        vertical.add(panes, weight=3)

        query_frame = ttk.LabelFrame(panes, text="SQL query", padding=5)
        self.query_text = tk.Text(query_frame, wrap=tk.NONE, font=("Courier New", 11), undo=True, width=40)
        self.query_text.insert("1.0", SAMPLE_QUERY)
        self.query_text.pack(fill=tk.BOTH, expand=True)
        self.compare_button = ttk.Button(query_frame, text="Compare plans", command=self.on_compare)
        self.compare_button.pack(fill=tk.X, pady=(5, 0))
        panes.add(query_frame, weight=1)

        plans_frame = ttk.Frame(panes)
        legend = ttk.Frame(plans_frame)
        legend.pack(fill=tk.X, pady=(0, 4))
        for colour, label in STATUS_COLOURS.values():
            tk.Label(legend, text=f"  {label}  ", bg=colour, fg="#111827").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(legend, text="Click a step to see its counterpart and why they differ.").pack(side=tk.LEFT,
                                                                                              padx=(8, 0))
        tabs = ttk.Notebook(plans_frame)
        tabs.pack(fill=tk.BOTH, expand=True)

        trees = ttk.PanedWindow(tabs, orient=tk.HORIZONTAL)
        self.postgres_canvas = PlanCanvas(trees, "PostgreSQL plan", self.on_node_selected)
        self.ai_canvas = PlanCanvas(trees, "AI agent plan", self.on_node_selected)
        trees.add(self.postgres_canvas, weight=1)
        trees.add(self.ai_canvas, weight=1)
        tabs.add(trees, text="Tree view")

        texts = ttk.PanedWindow(tabs, orient=tk.HORIZONTAL)
        self.postgres_view = self._make_text_view(texts, "PostgreSQL plan")
        self.ai_view = self._make_text_view(texts, "AI agent plan")
        tabs.add(texts, text="Text view")
        panes.add(plans_frame, weight=3)

        bottom = ttk.PanedWindow(vertical, orient=tk.HORIZONTAL)
        vertical.add(bottom, weight=1)
        self.report_view = self._make_text_view(bottom, "Comparison", wrap=tk.WORD, font_size=11, weight=3)
        self.report_view.tag_configure("big", font=("Courier New", 13, "bold"))
        self.details_view = self._make_text_view(bottom, "Selected step", wrap=tk.WORD, font_size=11, weight=2)
        self._set_text(self.details_view, [("Click a step in either tree to see its details here.", None)])

    def _make_text_view(self, parent, title, wrap=tk.NONE, font_size=10, weight=1) -> tk.Text:
        """A scrollable, read-only text panel.

        Its tags cover headings, warnings, status background colours (for plan lines) and
        coloured markers (for the explanation panels, whose text itself stays plain).
        """
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        text = tk.Text(frame, wrap=wrap, font=("Courier New", font_size), state=tk.DISABLED, height=10)
        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=y_scroll.set)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        if wrap == tk.NONE:
            x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
            text.configure(xscrollcommand=x_scroll.set)
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        text.pack(fill=tk.BOTH, expand=True)
        text.tag_configure("heading", font=("Courier New", font_size, "bold"))
        text.tag_configure("warning", foreground="#b45309")
        text.tag_configure("error", foreground="#b91c1c")
        for status, (colour, _) in STATUS_COLOURS.items():
            text.tag_configure(status.value, background=colour, foreground="#111827")
            text.tag_configure(f"marker_{status.value}", foreground=PlanCanvas.BORDERS[status])
        parent.add(frame, weight=weight)
        return text

    def _build_status_bar(self):
        """One line of status text along the bottom of the window."""
        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 4), anchor=tk.W).pack(fill=tk.X)


    def refresh_databases(self):
        """Fill the Database dropdown with the databases on the PostgreSQL server."""
        try:
            with DatabaseConnection.from_config(self.config) as db:
                self.database_box["values"] = db.list_databases()
            self.status_var.set("Connected to PostgreSQL")
        except PlanRetrievalError as e:
            self.status_var.set("Not connected - check config.json")
            messagebox.showerror("Connection error", str(e))

    def on_provider_changed(self, _event=None):
        """Suggest the default model when the user switches provider."""
        self.model_var.set(DEFAULT_MODELS.get(self.provider_var.get(), ""))

    def _current_config(self) -> dict:
        """The loaded config, overridden by what is selected in the window."""
        config = copy.deepcopy(self.config)
        config["database"]["dbname"] = self.database_var.get()
        config["ai"].update(provider=self.provider_var.get(), model=self.model_var.get().strip())
        return config

    def on_compare(self):
        """Clear the panels, then fetch and compare both plans on a background thread."""
        query = self.query_text.get("1.0", tk.END)
        self.compare_button.state(["disabled"])
        self.status_var.set("Retrieving PostgreSQL plan...")
        self.postgres_plan = self.ai_plan = self.comparison = None
        for view in (self.postgres_view, self.ai_view, self.report_view, self.details_view):
            self._set_text(view, [])
        self.postgres_canvas.show_message("Retrieving PostgreSQL plan...")
        self.ai_canvas.show_message("Waiting for the AI agent...")
        worker_args = (query, self._current_config())
        threading.Thread(target=self._compare_worker, args=worker_args, daemon=True).start()
        self.root.after(100, self._poll_results)

    def _compare_worker(self, query: str, config: dict):
        """Fetch both plans and compare them. Runs on a background thread; reports through result_queue."""
        try:
            with DatabaseConnection.from_config(config) as db:
                postgres_plan = db.get_postgres_plan(query)
                self.result_queue.put(("postgres", postgres_plan))
                schema = db.get_schema_info()
        except PlanRetrievalError as e:
            self.result_queue.put(("fatal", str(e)))
            return
        except Exception as e:
            self.result_queue.put(("fatal", f"Unexpected error: {e!r}"))
            return

        try:
            ai_plan = get_ai_plan(query, schema, config, expected_tables=postgres_plan.root.base_relations())
            self.result_queue.put(("ai", ai_plan))
            self.result_queue.put(("comparison", compare_plans(postgres_plan, ai_plan)))
        except PlanRetrievalError as e:
            self.result_queue.put(("ai_error", str(e)))
        except Exception as e:
            self.result_queue.put(("ai_error", f"Unexpected error: {e!r}"))
        self.result_queue.put(("done", None))

    def _poll_results(self):
        """Handle (on the main thread) whatever the background worker has reported so far."""
        while True:
            try:
                kind, payload = self.result_queue.get_nowait()
            except queue.Empty:
                self.root.after(100, self._poll_results)
                return

            if kind == "postgres":
                self.postgres_plan = payload
                self._set_text(self.postgres_view, self._plan_parts(payload))
                self.postgres_canvas.show(payload)
                self.status_var.set(f"PostgreSQL plan retrieved ({payload.root.size()} nodes). Asking the AI...")
            elif kind == "ai":
                self.ai_plan = payload
                self._set_text(self.ai_view, self._plan_parts(payload))
                self.ai_canvas.show(payload)
                self.status_var.set("AI plan retrieved. Comparing...")
            elif kind == "comparison":
                self._show_comparison(payload)
            elif kind == "ai_error":
                self._set_text(self.ai_view, [("Could not get the AI plan:\n\n", "heading"), (payload, "error")])
                self.ai_canvas.show_message("Could not get the AI plan:\n\n" + payload)
                self.status_var.set("PostgreSQL plan retrieved, but the AI plan failed (see right panel)")
            elif kind == "fatal":
                self.status_var.set("Error")
                self.postgres_canvas.show_message("No plan.")
                self.ai_canvas.show_message("No plan.")
                self.compare_button.state(["!disabled"])
                messagebox.showerror("Could not compare plans", payload)
                return
            elif kind == "done":
                self.compare_button.state(["!disabled"])
                return

    def _plan_parts(self, plan, node_status=None) -> list:
        """(text, tag) pieces for a plan view; each node's lines are coloured by its status."""
        parts = []
        if plan.source != "PostgreSQL":
            parts.append((f"Source: {plan.source}\n\n", "heading"))
        for node, line in plan.root.to_lines():
            status = (node_status or {}).get(id(node))
            parts.append((line, status.value if status else None))
            parts.append(("\n", None))
        if plan.reasoning:
            parts += [("\nAI reasoning:\n", "heading"), (plan.reasoning + "\n", None)]
        if plan.warnings:
            parts.append(("\nValidation warnings:\n", "heading"))
            parts += [(f"  - {w}\n", "error" if w in plan.serious_warnings else "warning")
                      for w in plan.warnings]
        return parts

    def _show_comparison(self, result):
        """Show a finished comparison."""
        self.comparison = result
        self._set_text(self.postgres_view, self._plan_parts(result.postgres_plan, result.node_status))
        self._set_text(self.ai_view, self._plan_parts(result.ai_plan, result.node_status))
        self.postgres_canvas.show(result.postgres_plan, result.node_status)
        serious = len(result.ai_plan.serious_warnings)
        minor = len(result.ai_plan.warnings) - serious
        note = " · ".join(text for count, text in ((serious, f"⚠ {serious} serious"), (minor, f"{minor} minor"))
                          if count)
        self.ai_canvas.show(result.ai_plan, result.node_status, note=f"{note} (Text view)" if note else "")

        breakdown = result.summary().split("  -  ", 1)[-1]
        parts = [(f"Difference rate: {result.difference_rate:.0%}\n", "big"), (breakdown + "\n", None)]
        differences = [(p.explanation, p.status) for p in result.pairs
                       if p.status is not MatchStatus.SAME and p.explanation]
        sections = [
            ("Differences", differences),
            ("Similarities", [(text, MatchStatus.SAME) for text in result.similarities]),
            ("Notes (not counted in the difference rate)", [(text, None) for text in result.notes]),
        ]
        for heading, items in sections:
            if items:
                parts.append((f"\n{heading}:\n", "heading"))
                for text, status in items:
                    marker = ("  ● ", f"marker_{status.value}") if status else ("  - ", None)
                    parts += [marker, (text, None), ("\n", None)]
        if not result.explanations:
            parts.append(("\nThe two plans are the same after normalisation.\n", "heading"))
        self._set_text(self.report_view, parts)
        self.status_var.set(f"Done - difference rate {result.difference_rate:.0%}")

    def on_node_selected(self, node, canvas: PlanCanvas):
        """A step was clicked in one tree: highlight its counterpart and explain the pair."""
        other = self.ai_canvas if canvas is self.postgres_canvas else self.postgres_canvas
        side, other_side = ("PostgreSQL", "AI") if canvas is self.postgres_canvas else ("AI", "PostgreSQL")
        result = self.comparison
        partner = result.partner.get(id(node)) if result else None
        if partner is not None:
            other.select(partner)
        else:
            other.clear_selection()

        parts = [(f"{side} plan: {node.label()}\n", "heading")]
        status = result.node_status.get(id(node)) if result else None
        if status is not None:
            parts += [("● ", f"marker_{status.value}"), (f"{STATUS_COLOURS[status][1]}\n", None)]
        estimates = []
        if node.estimated_rows is not None:
            estimates.append(f"estimated rows {node.estimated_rows:,.0f}")
        if node.estimated_cost is not None:
            estimates.append(f"estimated cost {node.estimated_cost:,.2f}")
        if estimates:
            parts.append((", ".join(estimates) + "\n", None))
        for key, value in list(node.conditions.items()) + [(k, v) for k, v in node.attributes.items()
                                                           if k in ("Sort Key", "Group Key", "Strategy")]:
            parts.append((f"{key}: {', '.join(value) if isinstance(value, list) else value}\n", None))

        if result is not None:
            if partner is not None:
                parts.append((f"\nCounterpart in the {other_side} plan: {partner.label()}\n", "heading"))
            else:
                parts.append((f"\nNo counterpart in the {other_side} plan.\n", "heading"))
            pair = result.node_pair.get(id(node))
            if pair is not None:
                if pair.explanation:
                    parts.append((pair.explanation + "\n", None))
                logical = pair.pg if canvas is self.postgres_canvas else pair.ai
                if logical is not None and node is not logical.plan_node:
                    parts.append((f"\n({node.node_type} is bookkeeping, so it was compared together with "
                                  f"{logical.plan_node.label()}.)\n", None))
        self._set_text(self.details_view, parts)

    @staticmethod
    def _set_text(widget: tk.Text, parts: list):
        """Replace a read-only text widget's content with (text, tag) pieces."""
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        for text, tag in parts:
            widget.insert(tk.END, text, tag or ())
        widget.configure(state=tk.DISABLED)


def launch_app(config: dict):
    """Create the main window and start the Tkinter event loop."""
    enable_high_dpi()
    root = tk.Tk()
    PlanComparisonApp(root, config)
    root.mainloop()
