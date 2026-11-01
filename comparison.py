"""comparison.py - Comparing two query execution plans.

Takes two PlanNode trees (one from PostgreSQL, one from the AI agent, both
produced by preprocessing.py) and works out where they agree and differ.

Algorithm:
  1. Normalise each tree into a "logical" tree, so that PostgreSQL bookkeeping
     doesn't count as a difference: Gather / Gather Merge are dropped,
     Partial + Finalize Aggregate become one Aggregate, a Hash node is folded
     into its Hash Join, Bitmap Index Scans into their Bitmap Heap Scan, and
     Materialize / Memoize into the node they cache.
  2. Give every logical node a signature: (category, multiset of base tables
     beneath it), e.g. ('join', ('nation', 'region')). Nodes with the same
     signature in both trees play the same role, so they are matched
     (hash lookup, in two passes - see match_nodes).
  3. Classify each pair: SAME (same method), CHANGED (same role, different
     method, e.g. Seq Scan vs Index Scan) or ONLY_IN_ONE (no counterpart,
     e.g. a join that only exists because of a different join order).
  4. Explain every difference in plain English from rule-based templates
     filled with values from the plans, and list the similarities.
  5. Difference rate = 1 - 2 * |SAME pairs| / (|PG logical nodes| + |AI logical nodes|),
     i.e. one minus the Dice similarity of the two normalised trees:
     0 = identical, 1 = nothing in common.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from preprocessing import PlanNode, QueryPlan


class MatchStatus(Enum):
    """How a step compares with the other plan.

    SAME (shown green):         matched and done the same way.
    CHANGED (shown amber):      matched, but done with a different method.
    ONLY_IN_ONE (shown red):    no counterpart in the other plan.
    """
    SAME = "same"
    CHANGED = "changed"
    ONLY_IN_ONE = "only_in_one"


SCAN_NODES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Tid Scan",
              "Sample Scan", "Function Scan", "Values Scan", "CTE Scan", "WorkTable Scan"}
JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}
CATEGORY_OF = {
    **{t: "scan" for t in SCAN_NODES},
    **{t: "join" for t in JOIN_NODES},
    "Aggregate": "aggregate", "Group": "aggregate",
    "Sort": "sort", "Incremental Sort": "sort",
    "Limit": "limit", "Unique": "distinct", "WindowAgg": "window",
    "Append": "set operation", "Merge Append": "set operation", "SetOp": "set operation",
    "Recursive Union": "set operation",
}
PARALLEL_WRAPPERS = {"Gather", "Gather Merge"}
CACHING_HELPERS = {"Materialize", "Memoize"}
BITMAP_INDEX_NODES = {"Bitmap Index Scan", "BitmapAnd", "BitmapOr"}
SUBPLAN_ROLES = {"InitPlan", "SubPlan"}


@dataclass(eq=False)
class LogicalNode:
    """One step of a plan after normalisation; remembers the original nodes it covers.

    Attributes:
        category:   scan / join / aggregate / sort / limit / distinct / window / set operation / other
        method:     how the step is done, e.g. 'Index Scan', 'Hash Join', or 'Hashed' for an aggregate
        plan_node:  the main original PlanNode
        members:    every original PlanNode folded into this step (e.g. a Gather or Hash node),
                    so all of them can be coloured with this step's status
        indexes:    index names used (a bitmap scan can combine several)
        join_type:  Inner / Left / Right / Full / Semi / Anti / Right Semi / Right Anti
        partial:    True for a Partial Aggregate not (yet) merged with its Finalize step
        parallel:   ran under a Gather, or was parallel-aware
        via_gather: a Gather was folded directly above this step
        helpers:    folded Materialize / Memoize nodes (caching on the inner side of a nested loop)
        is_subplan: a separately executed sub-query (InitPlan / SubPlan)
        tables:     sorted multiset of the base tables in this subtree - part of the signature
    """

    category: str
    method: str
    plan_node: PlanNode
    members: list = field(default_factory=list)
    children: list = field(default_factory=list)
    relation: Optional[str] = None
    alias: Optional[str] = None
    indexes: tuple = ()
    join_type: str = "Inner"
    partial: bool = False
    parallel: bool = False
    via_gather: bool = False
    helpers: list = field(default_factory=list)
    is_subplan: bool = False
    tables: tuple = ()

    def walk(self):
        """Post-order (children first), so matching proceeds bottom-up."""
        for child in self.children:
            yield from child.walk()
        yield self

    @property
    def signature(self) -> tuple:
        """(category, tables): steps with equal signatures play the same role in both plans."""
        return self.category, self.tables

    def describe(self) -> str:
        """Human-readable description of the method, e.g. 'an index scan using orders_pkey'."""
        idx = ", ".join(self.indexes) or "an unnamed index"
        if self.category == "scan":
            return {
                "Seq Scan": "a sequential scan (reads the whole table)",
                "Index Scan": f"an index scan using {idx}",
                "Index Only Scan": f"an index-only scan using {idx}",
                "Bitmap Scan": f"a bitmap scan using {idx}",
            }.get(self.method, self.method)
        if self.category == "join":
            text = f"a {self.method}"
            return text + (f" ({self.join_type.lower()} join)" if self.join_type != "Inner" else "")
        if self.category == "aggregate":
            return {
                "Hashed": "hash aggregation (groups rows in a hash table)",
                "Sorted": "sorted aggregation (groups rows that arrive sorted)",
                "Plain": "a single aggregate over all rows",
                "Mixed": "mixed hash/sorted aggregation",
            }.get(self.method, self.method)
        return f"a {self.method}"


def _table_id(node: PlanNode) -> Optional[str]:
    """Identifier of the 'table' a leaf reads: a base table, CTE, function or VALUES list."""
    if isinstance(node.relation, str) and node.relation:
        return node.relation.lower()
    if node.attributes.get("CTE Name"):
        return f"cte:{node.attributes['CTE Name']}"
    if node.attributes.get("Function Name"):
        return f"function:{node.attributes['Function Name']}"
    if node.node_type == "Values Scan":
        return "values"
    return None


def normalise(node: PlanNode) -> LogicalNode:
    """Build the logical tree for one plan (see module docstring, step 1).

    Folding rules:
      - Gather / Gather Merge only collect rows from parallel workers, and Materialize /
        Memoize only cache the inner side of a nested loop: each is folded into its child.
      - A Hash node only builds its Hash Join's hash table, so it is folded into the join.
      - A Bitmap Heap Scan and the Bitmap Index Scan(s) beneath it become one 'Bitmap Scan'.
      - Partial Aggregate + Finalize Aggregate (with or without a Gather between them)
        become one Aggregate.
    """
    node_type = node.node_type
    is_subplan = node.attributes.get("Parent Relationship") in SUBPLAN_ROLES
    parallel = bool(node.attributes.get("Parallel Aware"))

    if len(node.children) == 1 and node_type in PARALLEL_WRAPPERS | CACHING_HELPERS | {"Hash"}:
        child = normalise(node.children[0])
        child.members.append(node)
        child.is_subplan = child.is_subplan or is_subplan
        if node_type in PARALLEL_WRAPPERS:
            child.parallel = child.via_gather = True
        elif node_type in CACHING_HELPERS:
            child.helpers.append(node_type)
        else:
            child.parallel = child.parallel or parallel
        return child

    category = CATEGORY_OF.get(node_type, "other")
    logical = LogicalNode(category=category, method=node_type, plan_node=node, members=[node],
                          relation=node.relation, alias=node.alias, parallel=parallel,
                          is_subplan=is_subplan)
    if node.join_type:
        logical.join_type = node.join_type

    children = list(node.children)
    if node_type == "Bitmap Heap Scan":
        logical.method = "Bitmap Scan"
        index_nodes = [n for c in children if c.node_type in BITMAP_INDEX_NODES for n in c.walk()]
        logical.members += index_nodes
        logical.indexes = tuple(sorted({n.index_name for n in index_nodes if n.index_name}))
        children = [c for c in children if c.node_type not in BITMAP_INDEX_NODES]
    elif node.index_name:
        logical.indexes = (node.index_name,)
    elif node_type == "Aggregate":
        logical.method = node.attributes.get("Strategy", "Plain")
        logical.partial = node.attributes.get("Partial Mode") == "Partial"
    elif node_type == "Group":
        logical.method = "Sorted"
    if category == "join":
        hash_nodes = [c for c in children if c.node_type == "Hash" and len(c.children) == 1]
        logical.members += hash_nodes
        children = [c.children[0] if c in hash_nodes else c for c in children]

    logical.children = [normalise(c) for c in children]

    if (logical.category == "aggregate" and len(logical.children) == 1
            and logical.children[0].category == "aggregate"
            and (logical.children[0].partial or logical.children[0].via_gather)):
        inner = logical.children.pop()
        logical.members += inner.members
        logical.children = inner.children
        logical.parallel = logical.parallel or inner.parallel

    tables = [_table_id(node)] if category == "scan" and _table_id(node) else []
    for child in logical.children:
        if not child.is_subplan:
            tables += child.tables
    logical.tables = tuple(sorted(tables))
    return logical


@dataclass
class NodePair:
    """A matched pair of logical nodes, or a node with no counterpart (other side None)."""

    pg: Optional[LogicalNode]
    ai: Optional[LogicalNode]
    status: MatchStatus
    explanation: str = ""


def match_nodes(pg_root: LogicalNode, ai_root: LogicalNode) -> list:
    """Pair up nodes with equal signatures; leftovers become ONLY_IN_ONE pairs.

    Matching runs in two passes, so an early weak match can't take a candidate that a
    later node fits better: pass 1 only accepts candidates in exactly the same position
    (the same kinds of node directly above and below), pass 2 matches the rest. With
    several candidates (a self-join, two Sorts over the same tables) the same alias wins,
    then the same position, then the same method. Pairs are returned in bottom-up
    PostgreSQL order, followed by the steps only the AI has.
    """
    ai_by_signature = defaultdict(list)
    for node in ai_root.walk():
        ai_by_signature[node.signature].append(node)

    context = {}
    for root in (pg_root, ai_root):
        context[id(root)] = (None, tuple(c.category for c in root.children))
        for node in root.walk():
            for child in node.children:
                context[id(child)] = (node.category, tuple(c.category for c in child.children))

    matched = {}
    pending = list(pg_root.walk())
    for strict in (True, False):
        unmatched = []
        for pg_node in pending:
            candidates = ai_by_signature.get(pg_node.signature, [])
            if strict:
                usable = [a for a in candidates if context[id(a)] == context[id(pg_node)]]
            else:
                usable = candidates
            if not usable:
                unmatched.append(pg_node)
                continue
            pg_above, pg_below = context[id(pg_node)]
            best = max(usable, key=lambda a: (pg_node.alias is not None and a.alias == pg_node.alias,
                                              context[id(a)][0] == pg_above,
                                              context[id(a)][1] == pg_below,
                                              a.method == pg_node.method))
            candidates.remove(best)
            matched[id(pg_node)] = best
        pending = unmatched

    pairs = []
    for pg_node in pg_root.walk():
        ai_node = matched.get(id(pg_node))
        if ai_node is None:
            pairs.append(NodePair(pg_node, None, MatchStatus.ONLY_IN_ONE))
        else:
            pairs.append(NodePair(pg_node, ai_node, _classify(pg_node, ai_node)))
    for leftovers in ai_by_signature.values():
        pairs += [NodePair(None, ai_node, MatchStatus.ONLY_IN_ONE) for ai_node in leftovers]
    return pairs


def _sides_swapped(pg: LogicalNode, ai: LogicalNode) -> bool:
    """True if both joins combine the same inputs but use them as outer/inner the other way round."""
    if len(pg.children) != 2 or len(ai.children) != 2:
        return False
    return (pg.children[0].tables == ai.children[1].tables
            and pg.children[1].tables == ai.children[0].tables
            and pg.children[0].tables != pg.children[1].tables)


_RIGHT_TO_LEFT = {"Right": "Left", "Right Anti": "Anti", "Right Semi": "Semi"}


def _logical_join(node: LogicalNode) -> tuple:
    """(join type, tables whose rows the join keeps), regardless of which input is outer/inner.

    A "Right" join is the matching left-hand join written with its inputs the other way
    round: a Right Anti join with (A outer, B inner) keeps the unmatched rows of B, exactly
    like an Anti join with (B outer, A inner). PostgreSQL picks the Right form when it
    prefers to hash / loop over the other input, so both forms map to the same result here.
    """
    kind = _RIGHT_TO_LEFT.get(node.join_type, node.join_type)
    if kind in ("Inner", "Full") or len(node.children) != 2:
        return kind, None
    kept = node.children[1] if node.join_type in _RIGHT_TO_LEFT else node.children[0]
    return kind, kept.tables


def _classify(pg: LogicalNode, ai: LogicalNode) -> MatchStatus:
    """SAME if both steps use the same method, otherwise CHANGED.

    Index scans must also use the same indexes; joins must also be the same logical join
    (see _logical_join) with their inputs on the same sides.
    """
    same = pg.method == ai.method
    if pg.category == "scan" and pg.method != "Seq Scan":
        same = same and set(pg.indexes) == set(ai.indexes)
    if pg.category == "join":
        same = same and _logical_join(pg) == _logical_join(ai) and not _sides_swapped(pg, ai)
    return MatchStatus.SAME if same else MatchStatus.CHANGED


JOIN_HINTS = {
    "Hash Join": "builds a hash table on one input and probes it with the other; suits large inputs",
    "Nested Loop": "looks up matches in the inner input once per outer row; suits a small outer input "
                   "or an index on the inner side",
    "Merge Join": "walks two inputs that are sorted on the join key; suits inputs that are already sorted",
}


def _list(items) -> str:
    """Join names as 'a', 'a and b' or 'a, b and c', dropping duplicates and keeping order."""
    items = list(dict.fromkeys(items))
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _a(word: str) -> str:
    """'inner' -> 'an inner', 'semi' -> 'a semi'."""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _rows(node: LogicalNode) -> str:
    """PostgreSQL's row estimate for a step, as text (per worker for parallel steps)."""
    rows = node.plan_node.estimated_rows
    if rows is None:
        return "an unknown number of"
    return f"about {rows:,.0f}" + (" (per parallel worker)" if node.parallel else "")


def _explain_scan(pg: LogicalNode, ai: LogicalNode, parent: Optional[LogicalNode]) -> str:
    """Explain why two scans of the same table differ, using PostgreSQL's row estimate."""
    table = pg.relation or _list(pg.tables)
    text = f"Table {table}: PostgreSQL uses {pg.describe()}, the AI uses {ai.describe()}. "
    has_filter = any(k in pg.plan_node.conditions for k in ("Filter", "Index Cond", "Recheck Cond"))
    inner_of_loop = parent is not None and parent.method == "Nested Loop" and parent.children[-1] is pg

    if pg.method == "Index Only Scan":
        index = pg.indexes[0] if pg.indexes else "the index"
        text += (f"Every column this step needs is stored in {index}, so PostgreSQL reads only that index "
                 "and never visits the table itself; the index is much smaller than the table, so this is "
                 f"cheaper than reading the table ({_rows(pg)} rows estimated).")
        if inner_of_loop:
            text += " It is looked up once for each row coming from the nested loop above."
        if ai.method == "Index Scan":
            text += " The AI uses an index too, but also fetches every matching row from the table."
    elif ai.method == "Index Only Scan":
        text += (f"The AI assumes every column needed is stored in {ai.indexes[0] if ai.indexes else 'the index'}, "
                 f"so it never reads the table. PostgreSQL does read the table here (it estimates {_rows(pg)} "
                 "rows), because the index doesn't hold all the columns it needs or a plain read is cheaper.")
    elif pg.method == "Seq Scan" and ai.method != "Seq Scan":
        if not has_filter:
            text += (f"The query needs every row of {table} (no filter applies to it here), so PostgreSQL "
                     "reads it sequentially; going through an index would add random lookups for no gain.")
        else:
            text += (f"PostgreSQL expects the filter to keep {_rows(pg)} rows, a large share of the table, "
                     "so one sequential pass is cheaper than that many index lookups. The AI assumed the "
                     "index would be selective enough to pay off.")
    elif pg.method != "Seq Scan" and ai.method == "Seq Scan":
        if inner_of_loop:
            text += (f"PostgreSQL probes {pg.indexes[0] if pg.indexes else 'the index'} once for each row "
                     "coming from the nested loop above, which is far cheaper than re-reading the whole "
                     "table each time; the AI reads the whole table instead.")
        else:
            text += (f"PostgreSQL expects only {_rows(pg)} matching rows, so it uses the index to avoid "
                     "reading the whole table; the AI reads the entire table.")
    elif {pg.method, ai.method} == {"Index Scan", "Bitmap Scan"}:
        text += ("A bitmap scan first collects all matching row locations and then reads the table in "
                 "physical order, which suits a moderate number of matches; a plain index scan suits very "
                 f"few matches. PostgreSQL estimates {_rows(pg)} rows here.")
    elif pg.method == ai.method:
        text += "The scan type is the same but a different index is used."
    return text


def _explain_join(pg: LogicalNode, ai: LogicalNode) -> str:
    """Explain why two joins over the same tables differ."""
    tables = _list(pg.tables)
    parts = []
    if pg.method != ai.method:
        parts.append(f"To join {tables}, PostgreSQL uses a {pg.method} ({JOIN_HINTS.get(pg.method, '')}), "
                     f"while the AI uses a {ai.method} ({JOIN_HINTS.get(ai.method, '')}). PostgreSQL "
                     f"estimates the outer input at {_rows(pg.children[0])} rows." if pg.children else "")
    (pg_kind, pg_kept), (ai_kind, ai_kept) = _logical_join(pg), _logical_join(ai)
    if pg_kind != ai_kind:
        parts.append(f"PostgreSQL performs this join as {_a(pg.join_type.lower())} join, the AI as "
                     f"{_a(ai.join_type.lower())} join (EXISTS / IN usually become semi joins; NOT EXISTS "
                     "and LEFT JOIN ... IS NULL can become anti joins).")
    elif pg_kept is not None and ai_kept is not None and pg_kept != ai_kept:
        parts.append(f"Both are {pg_kind.lower()} joins, but they keep the rows of different tables: "
                     f"PostgreSQL keeps {_list(pg_kept)}, the AI keeps {_list(ai_kept)}, so they don't "
                     "compute the same result.")
    elif pg.join_type != ai.join_type:
        parts.append(f"Both perform the same {pg_kind.lower()} join (keeping rows of {_list(pg_kept)}); "
                     f"PostgreSQL's \"{pg.join_type}\" and the AI's \"{ai.join_type}\" only differ because "
                     "the two inputs are in the opposite order.")
    if _sides_swapped(pg, ai):
        mine, theirs = pg.children[1], pg.children[0]
        if pg.method == "Hash Join":
            text = (f"PostgreSQL builds its hash table on {_list(mine.tables)} (estimated {_rows(mine)} rows), "
                    f"the AI on {_list(theirs.tables)} (PostgreSQL estimates {_rows(theirs)} rows)")
            smaller = (mine.plan_node.estimated_rows or 0) < (theirs.plan_node.estimated_rows or 0)
            parts.append(text + ("; hashing the smaller input is cheaper." if smaller else "."))
        else:
            parts.append(f"The inputs are swapped: PostgreSQL uses {_list(mine.tables)} as the inner "
                         f"(repeated) input, the AI uses {_list(theirs.tables)}.")
    if not any(parts):
        parts.append(f"Both plans join {tables} here, but their join steps are built from different "
                     f"inputs (PostgreSQL: {len(pg.children)}, AI: {len(ai.children)}).")
    return " ".join(p for p in parts if p)


def _explain_aggregate(pg: LogicalNode, ai: LogicalNode) -> str:
    """Explain a different grouping strategy (hash, sorted or plain aggregation)."""
    text = f"For grouping, PostgreSQL uses {pg.describe()}, the AI uses {ai.describe()}. "
    if {pg.method, ai.method} == {"Hashed", "Sorted"}:
        text += ("Hash aggregation avoids sorting but must hold every group in memory (work_mem); "
                 "sorted aggregation is preferred when the input is already sorted or there are many groups.")
    return text


def _explain_unmatched(node: LogicalNode, parent: Optional[LogicalNode], who: str, other: str,
                       other_tables: set) -> str:
    """Explain a step that exists in only one plan."""
    tables = _list(node.tables) if node.tables else "no tables"
    if node.category == "join":
        missing = [t for t in node.tables if t not in other_tables]
        if missing:
            return (f"{who} joins {tables} at this step; {other}'s plan cannot build this result because "
                    f"it never reads {_list(missing)}.")
        return (f"{who} joins {tables} at this step; {other}'s plan never builds this intermediate result "
                "because it joins the tables in a different order.")
    if node.category == "scan":
        return f"{who} reads {tables}, but {other}'s plan does not read it at all."
    feeding = f" to feed the {_step_name(parent)} step above it" if parent else ""
    if node.category == "sort":
        return f"{who} sorts the rows over {tables}{feeding}; {other}'s plan has no sort here."
    return f"{who} has a {node.method} step over {tables}{feeding} that {other}'s plan lacks."


def _step_name(node: LogicalNode) -> str:
    """Short name for a step, e.g. 'Hash Join' or 'sorted aggregation'."""
    return f"{node.method.lower()} aggregation" if node.category == "aggregate" else node.method


def _explain_same(pg: LogicalNode) -> str:
    """One-line description of a step both plans perform the same way."""
    tables = _list(pg.tables) if pg.tables else "the input"
    if pg.category == "scan":
        return f"Both read {pg.relation or tables} with {pg.describe()}."
    if pg.category == "join":
        return f"Both join {tables} with {pg.describe()}."
    if pg.category == "aggregate":
        return f"Both group the rows with {pg.describe()}."
    return f"Both apply a {pg.method} step over {tables}."


@dataclass
class ComparisonResult:
    """Everything the GUI needs to display a comparison.

    Attributes:
        node_status:     id(PlanNode) -> MatchStatus, for every node of both trees
        partner:         id(PlanNode) -> the matching PlanNode in the other tree
        node_pair:       id(PlanNode) -> the NodePair (with its explanation) the node belongs to
        pairs:           every NodePair
        explanations:    the differences in plain English
        similarities:    the steps both plans perform the same way
        notes:           minor differences that were normalised away and don't affect the score
        difference_rate: 0 = identical, 1 = nothing in common
        counts:          step counts: pg_nodes, ai_nodes, same, changed, only_pg, only_ai
        flags:           which kinds of difference occurred (scan_method, join_method, join_order, ...)
    """

    postgres_plan: QueryPlan
    ai_plan: QueryPlan
    node_status: dict = field(default_factory=dict)
    partner: dict = field(default_factory=dict)
    node_pair: dict = field(default_factory=dict)
    pairs: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    similarities: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    difference_rate: float = 0.0
    counts: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary: the difference rate and the step counts behind it."""
        c = self.counts
        return (f"Difference rate: {self.difference_rate:.0%}  -  {c['same']} identical steps, "
                f"{c['changed']} changed, {c['only_pg']} only in PostgreSQL, {c['only_ai']} only in the AI plan "
                f"({c['pg_nodes']} PostgreSQL vs {c['ai_nodes']} AI steps after normalisation)")


def compare_plans(postgres_plan: QueryPlan, ai_plan: QueryPlan) -> ComparisonResult:
    """Compare two plans and explain their differences."""
    pg_root, ai_root = normalise(postgres_plan.root), normalise(ai_plan.root)
    parents = {}
    for root in (pg_root, ai_root):
        for node in root.walk():
            for child in node.children:
                parents[id(child)] = node

    pg_tables = {t for n in pg_root.walk() for t in n.tables}
    ai_tables = {t for n in ai_root.walk() for t in n.tables}

    result = ComparisonResult(postgres_plan, ai_plan)
    result.pairs = match_nodes(pg_root, ai_root)

    for pair in result.pairs:
        pg, ai = pair.pg, pair.ai
        if pair.status is MatchStatus.SAME:
            pair.explanation = _explain_same(pg)
            result.similarities.append(pair.explanation)
        elif pair.status is MatchStatus.CHANGED:
            if pg.category == "scan":
                pair.explanation = _explain_scan(pg, ai, parents.get(id(pg)))
            elif pg.category == "join":
                pair.explanation = _explain_join(pg, ai)
            elif pg.category == "aggregate":
                pair.explanation = _explain_aggregate(pg, ai)
            else:
                pair.explanation = f"Over {_list(pg.tables)}, PostgreSQL uses {pg.describe()}, the AI {ai.describe()}."
            result.explanations.append(pair.explanation)
        elif pg is not None:
            pair.explanation = _explain_unmatched(pg, parents.get(id(pg)), "PostgreSQL", "the AI", ai_tables)
            result.explanations.append(pair.explanation)
        else:
            pair.explanation = _explain_unmatched(ai, parents.get(id(ai)), "The AI", "PostgreSQL", pg_tables)
            result.explanations.append(pair.explanation)

        for mine, theirs in ((pg, ai), (ai, pg)):
            if mine is None:
                continue
            for member in mine.members:
                result.node_status[id(member)] = pair.status
                result.node_pair[id(member)] = pair
                if theirs is not None:
                    result.partner[id(member)] = theirs.plan_node

    parallel_differs = _add_notes(result, pg_root, ai_root)

    pg_count, ai_count = sum(1 for _ in pg_root.walk()), sum(1 for _ in ai_root.walk())
    same = sum(p.status is MatchStatus.SAME for p in result.pairs)
    result.difference_rate = 1 - 2 * same / (pg_count + ai_count)
    result.counts = {
        "pg_nodes": pg_count, "ai_nodes": ai_count, "same": same,
        "changed": sum(p.status is MatchStatus.CHANGED for p in result.pairs),
        "only_pg": sum(p.status is MatchStatus.ONLY_IN_ONE and p.ai is None for p in result.pairs),
        "only_ai": sum(p.status is MatchStatus.ONLY_IN_ONE and p.pg is None for p in result.pairs),
    }

    def differs(category, *, unmatched_only=False):
        """True if a step of this category differs (with unmatched_only: has no counterpart)."""
        return any((p.pg or p.ai).category == category and p.status is not MatchStatus.SAME
                   and (not unmatched_only or p.status is MatchStatus.ONLY_IN_ONE) for p in result.pairs)

    result.flags = {
        "identical": result.difference_rate == 0,
        "scan_method": differs("scan"),
        "join_method": any(p.status is MatchStatus.CHANGED and p.pg.category == "join" for p in result.pairs),
        "join_order": differs("join", unmatched_only=True),
        "aggregation": differs("aggregate"),
        "sort": differs("sort"),
        "parallelism": parallel_differs,
    }
    return result


def _add_notes(result: ComparisonResult, pg_root: LogicalNode, ai_root: LogicalNode) -> bool:
    """Record minor differences that were normalised away (they don't count in the score).

    Returns whether the plans differ in their use of parallel workers.
    """
    pg_parallel = any(n.parallel for n in pg_root.walk())
    ai_parallel = any(n.parallel for n in ai_root.walk())
    if pg_parallel != ai_parallel:
        who, other = ("PostgreSQL", "the AI's") if pg_parallel else ("The AI", "PostgreSQL's")
        result.notes.append(f"{who} runs part of the plan with parallel workers (Gather); {other} plan is serial.")

    for pair in result.pairs:
        if pair.pg and pair.ai and sorted(pair.pg.helpers) != sorted(pair.ai.helpers):
            result.notes.append(
                f"Over {_list(pair.pg.tables)}, caching steps differ: PostgreSQL "
                f"{_list(pair.pg.helpers) if pair.pg.helpers else 'none'}, AI "
                f"{_list(pair.ai.helpers) if pair.ai.helpers else 'none'}.")

    pg_cost, ai_cost = result.postgres_plan.root.estimated_cost, result.ai_plan.root.estimated_cost
    if pg_cost is not None and ai_cost is not None:
        result.notes.append(f"Estimated total cost: PostgreSQL {pg_cost:,.0f}, the AI's own guess {ai_cost:,.0f} "
                            "(the AI's figure is not computed by PostgreSQL).")
    return pg_parallel != ai_parallel


if __name__ == "__main__":
    from preprocessing import DatabaseConnection, get_ai_plan, load_config

    test_queries = [
        "SELECT * FROM orders WHERE o_orderkey = 4242",
        """SELECT c_name, sum(o_totalprice) AS total_spent FROM customer, orders
           WHERE c_custkey = o_custkey AND o_orderdate >= date '1995-01-01'
           GROUP BY c_name ORDER BY total_spent DESC LIMIT 10""",
        """SELECT n_name, count(*) FROM nation n JOIN supplier s ON s.s_nationkey = n.n_nationkey
           JOIN region r ON n.n_regionkey = r.r_regionkey WHERE r.r_name = 'ASIA'
           AND EXISTS (SELECT 1 FROM partsupp ps WHERE ps.ps_suppkey = s.s_suppkey AND ps_availqty < 10)
           GROUP BY n_name ORDER BY 2 DESC""",
    ]
    cfg = load_config()
    with DatabaseConnection.from_config(cfg) as db:
        schema = db.get_schema_info()
        for sql in test_queries:
            print("=" * 78 + "\n" + " ".join(sql.split()))
            pg_plan = db.get_postgres_plan(sql)
            ai_plan = get_ai_plan(sql, schema, cfg, expected_tables=pg_plan.root.base_relations())
            comparison = compare_plans(pg_plan, ai_plan)
            print(comparison.summary())
            for heading, items in (("Differences", comparison.explanations),
                                   ("Similarities", comparison.similarities), ("Notes", comparison.notes)):
                if items:
                    print(f"\n{heading}:")
                    for item in items:
                        print(f"  - {item}")
            print()
