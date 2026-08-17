"""project-v1: build the thing the spec describes, not the thing the examples show.

Every other coding task here is one function with fixed cases, and the suite has saturated on
them — qwen3.8 scored 45/47 and gemma4 44/47, one task apart, which is inside noise. A
benchmark that cannot separate the two best models it measures has stopped earning its runtime.
It also measures the wrong unit: agents on this box write multi-part changes against prose
specs, and a model can be excellent at `mid_floor` and poor at "build this".

The design turns on one split. Each spec states a few REAL test cases inline, and the grader
also runs cases the model never saw. Passing the visible ones proves nothing except that the
model can copy an example; passing the hidden ones means it read the requirement the examples
only gesture at. The interesting result is the gap between them:

    visible 3/3, hidden 1/6   the model pattern-matched the examples
    visible 3/3, hidden 6/6   the model implemented the spec
    visible 0/3, hidden 0/6   it did not build, or misread entirely

That gap is the metric. A single percentage would average it away, which is why the two are
reported apart.

Three tiers, because "write code" is not one skill:

  build     from nothing, against a spec
  gap       the spec is deliberately silent on one case — does the model handle it safely and
            say so, or crash on it silently
  refactor  existing code plus a changed requirement, so it has to read before it writes

Two axes crossed over those tiers:

  language  named in the spec, or left to the model. The langpref suite already shows models
            have strong unprompted preferences; this asks whether obeying an explicit choice
            costs them correctness.
  length    a short spec fits any window. The long ones bury a requirement in the middle,
            where a model that skims the opening and the examples will miss it — the same
            depth effect the long-context ladder measures, on prose the model must act on
            rather than recall.

SECURITY: same contract as every other executable grader — model-generated code runs in a
separate process with a hard timeout, a scratch cwd and a stripped environment. That contains
accidents, not hostile code.
"""

# --- tier: build, language named, short spec --------------------------------------------------

_CALC_SPEC = '''Write a Python function `calc(expr: str) -> int` that evaluates an integer
arithmetic expression.

Requirements:
  1. Support + - * / and parentheses, with normal precedence.
  2. Division is integer division that truncates TOWARD ZERO, not floor division.
     So calc("-7/2") is -3, not -4.
  3. A unary minus is allowed anywhere a number is: calc("-(3+4)") is -7.
  4. Division by zero raises ValueError.
  5. Whitespace anywhere is ignored.
  6. The input is trusted to be well formed apart from division by zero. Do not use eval().

Examples that must work:
  calc("2+3*4")     == 14
  calc("(2+3)*4")   == 20
  calc("10/3")      == 3
'''

# --- tier: build, language named, LONG spec ---------------------------------------------------

_REPORT_SPEC = '''Write a Python function `report(rows: list[dict]) -> dict` that summarises
sales rows. Each row is a dict with keys: region (str), product (str), units (int),
unit_price (int, in whole cents), and status (str).

Requirements, in order:
  1. Ignore any row whose status is "void". Those rows do not exist for any purpose below.
  2. A row's revenue is units * unit_price.
  3. The result has a key "total" — the summed revenue of all non-void rows.
  4. The result has a key "by_region" — a dict of region to summed revenue.
  5. Regions with zero non-void rows do not appear in by_region at all.
  6. The result has a key "top_product" — the product with the highest summed revenue.
  7. Ties for top_product are broken alphabetically, earliest wins.
  8. The result has a key "count" — the number of non-void rows.
  9. Negative units are allowed; they represent returns and reduce revenue normally.
 10. A row with units == 0 counts toward "count" but contributes no revenue.
 11. The result has a key "regions" — the region names, sorted alphabetically.
 12. The result has a key "avg_order" — total divided by count, integer division truncating
     toward zero. If count is 0, avg_order is 0 rather than an error.
 13. A row whose status is "pending" IS included everywhere, exactly like a normal row.
 14. IMPORTANT: any row whose product name begins with an underscore is an internal test
     record. Exclude it from every calculation, exactly as if it were void.
 15. An empty input returns {"total": 0, "by_region": {}, "top_product": None, "count": 0,
     "regions": [], "avg_order": 0}.
 16. Do not mutate the input rows.
 17. The function must not raise on any input matching the shapes described here.

Examples that must work:
  report([]) == {"total": 0, "by_region": {}, "top_product": None, "count": 0,
                 "regions": [], "avg_order": 0}

  report([{"region": "north", "product": "bolt", "units": 3, "unit_price": 100,
           "status": "ok"}])
    == {"total": 300, "by_region": {"north": 300}, "top_product": "bolt", "count": 1,
        "regions": ["north"], "avg_order": 300}
'''

# --- tier: gap, language named, short spec ----------------------------------------------------

_MERGE_SPEC = '''Write a Python function `merge(a: dict, b: dict) -> dict` that merges two
configuration dictionaries.

Requirements:
  1. Keys present in only one input appear in the result unchanged.
  2. Where both inputs have a key and both values are dicts, merge them recursively.
  3. Where both inputs have a key and the values are not both dicts, b wins.
  4. Neither input may be mutated.

Examples that must work:
  merge({"a": 1}, {"b": 2})                    == {"a": 1, "b": 2}
  merge({"a": {"x": 1}}, {"a": {"y": 2}})      == {"a": {"x": 1, "y": 2}}
  merge({"a": 1}, {"a": 2})                    == {"a": 2}

Note: the spec above does not say what happens when a key's value is a list in both inputs.
Choose a behaviour, make it safe, and state your choice in a comment at the top of the file.
'''

# --- tier: refactor, language named, short spec ------------------------------------------------

_REFACTOR_SPEC = '''Here is an existing function. It is currently correct for its old
requirement, which was "return the total price in whole cents".

    def total(items):
        out = 0
        for it in items:
            out += it["qty"] * it["cents"]
        return out

The requirement has changed. Rewrite `total(items)` so that:
  1. It still returns whole cents as an int.
  2. Any item with a "discount" key has that percentage taken off ITS line total, rounded
     half up to the nearest cent. discount is an int from 0 to 100.
  3. Items with qty <= 0 are skipped entirely.
  4. An item missing "cents" is skipped rather than raising.
  5. The order of items does not affect the result.

Examples that must work:
  total([{"qty": 2, "cents": 500}])                    == 1000
  total([{"qty": 2, "cents": 500, "discount": 10}])    == 900
  total([{"qty": 0, "cents": 500}])                    == 0
'''

# --- tier: build, language NOT named -----------------------------------------------------------

_WINDOW_SPEC = '''Write a function `window_max(nums, k)` that returns a list of the maximum
value in every contiguous window of length k.

Requirements:
  1. For a list of n numbers and window k, the result has n - k + 1 entries.
  2. If k is larger than the list, return an empty list.
  3. If k is 0 or negative, return an empty list.
  4. The input list is not modified.
  5. Negative numbers work normally.

Examples that must work:
  window_max([1, 3, 2, 5, 4], 2)  -> [3, 3, 5, 5]
  window_max([1, 2], 5)           -> []

Use whichever language you consider best suited to this task. Say nothing but the code.
'''


def _case(args, expect, visible=False, label=None):
    """One graded case. `visible` means the spec shows this exact example, so passing it proves
    only that the model can copy — the hidden ones are what say it read the requirement."""
    return {"args": args, "expect": expect, "visible": visible,
            **({"label": label} if label else {})}


PROJECT_TASKS = [
    {
        "id": "proj_calc",
        "category": "project", "tier": "build", "lang": "python", "entry": "calc",
        "spec_size": "short", "lang_named": True,
        "prompt": _CALC_SPEC,
        "cases": [
            _case(["2+3*4"], 14, visible=True),
            _case(["(2+3)*4"], 20, visible=True),
            _case(["10/3"], 3, visible=True),
            # Requirement 2 — truncation toward zero is stated in prose and shown by no example.
            _case(["-7/2"], -3, label="truncates toward zero, not floor"),
            _case(["7/-2"], -3, label="truncates toward zero with the sign on the divisor"),
            # Requirement 3 — unary minus, stated but never demonstrated.
            _case(["-(3+4)"], -7, label="unary minus on a parenthesised group"),
            _case(["2*-3"], -6, label="unary minus after an operator"),
            # Requirement 5 — whitespace, stated in one line and shown nowhere.
            _case([" 2 + 3 * 4 "], 14, label="whitespace anywhere is ignored"),
        ],
    },
    {
        "id": "proj_report",
        "category": "project", "tier": "build", "lang": "python", "entry": "report",
        "spec_size": "long", "lang_named": True,
        "prompt": _REPORT_SPEC,
        "cases": [
            _case([[]], {"total": 0, "by_region": {}, "top_product": None, "count": 0,
                         "regions": [], "avg_order": 0}, visible=True),
            _case([[{"region": "north", "product": "bolt", "units": 3, "unit_price": 100,
                     "status": "ok"}]],
                  {"total": 300, "by_region": {"north": 300}, "top_product": "bolt",
                   "count": 1, "regions": ["north"], "avg_order": 300}, visible=True),
            # Requirement 14 sits two thirds of the way down a seventeen-point spec and is
            # demonstrated by no example. A model that skims the opening and the examples
            # cannot pass this one.
            _case([[{"region": "n", "product": "_probe", "units": 5, "unit_price": 100,
                     "status": "ok"},
                    {"region": "n", "product": "bolt", "units": 1, "unit_price": 200,
                     "status": "ok"}]],
                  {"total": 200, "by_region": {"n": 200}, "top_product": "bolt", "count": 1,
                   "regions": ["n"], "avg_order": 200},
                  label="requirement 14: underscore products are excluded like void rows"),
            _case([[{"region": "s", "product": "a", "units": 2, "unit_price": 100,
                     "status": "void"}]],
                  {"total": 0, "by_region": {}, "top_product": None, "count": 0,
                   "regions": [], "avg_order": 0},
                  label="requirement 5: a region with only void rows is absent, not zero"),
            _case([[{"region": "n", "product": "b", "units": 2, "unit_price": 100,
                     "status": "ok"},
                    {"region": "n", "product": "a", "units": 2, "unit_price": 100,
                     "status": "ok"}]],
                  {"total": 400, "by_region": {"n": 400}, "top_product": "a", "count": 2,
                   "regions": ["n"], "avg_order": 200},
                  label="requirement 7: a tie for top_product breaks alphabetically"),
            _case([[{"region": "n", "product": "b", "units": -1, "unit_price": 100,
                     "status": "ok"},
                    {"region": "n", "product": "b", "units": 3, "unit_price": 100,
                     "status": "pending"}]],
                  {"total": 200, "by_region": {"n": 200}, "top_product": "b", "count": 2,
                   "regions": ["n"], "avg_order": 100},
                  label="requirements 9 and 13: returns reduce revenue, pending counts"),
            _case([[{"region": "n", "product": "b", "units": 0, "unit_price": 100,
                     "status": "ok"}]],
                  {"total": 0, "by_region": {"n": 0}, "top_product": "b", "count": 1,
                   "regions": ["n"], "avg_order": 0},
                  label="requirements 10 and 12: a zero-unit row counts, avg_order does not divide by zero"),
        ],
    },
    {
        "id": "proj_merge_gap",
        "category": "project", "tier": "gap", "lang": "python", "entry": "merge",
        "spec_size": "short", "lang_named": True,
        "prompt": _MERGE_SPEC,
        "cases": [
            _case([{"a": 1}, {"b": 2}], {"a": 1, "b": 2}, visible=True),
            _case([{"a": {"x": 1}}, {"a": {"y": 2}}], {"a": {"x": 1, "y": 2}}, visible=True),
            _case([{"a": 1}, {"a": 2}], {"a": 2}, visible=True),
            # The gap: two lists under one key. The spec does not say. Any answer that does not
            # raise is acceptable, so the case asserts only that a value came back — what is
            # graded is that the model chose a behaviour rather than crashing on it.
            _case([{"a": [1]}, {"a": [2]}], {"a": [2]},
                  label="the unspecified case: b wins is the safe reading of requirement 3"),
            _case([{"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}],
                  {"a": {"b": {"c": 1, "d": 2}}}, label="recursion goes deeper than one level"),
        ],
    },
    {
        "id": "proj_refactor_total",
        "category": "project", "tier": "refactor", "lang": "python", "entry": "total",
        "spec_size": "short", "lang_named": True,
        "prompt": _REFACTOR_SPEC,
        "cases": [
            _case([[{"qty": 2, "cents": 500}]], 1000, visible=True),
            _case([[{"qty": 2, "cents": 500, "discount": 10}]], 900, visible=True),
            _case([[{"qty": 0, "cents": 500}]], 0, visible=True),
            _case([[{"qty": 1, "cents": 101, "discount": 50}]], 51,
                  label="rounds half UP, not to even: 50.5 becomes 51"),
            _case([[{"qty": 1}]], 0, label="an item missing cents is skipped, not an error"),
            _case([[{"qty": -2, "cents": 500}]], 0, label="negative qty is skipped"),
            _case([[{"qty": 1, "cents": 100, "discount": 100}]], 0,
                  label="a full discount is zero, not the undiscounted price"),
        ],
    },
    {
        "id": "proj_window_free",
        "category": "project", "tier": "build", "lang": "auto", "entry": "window_max",
        "spec_size": "short", "lang_named": False,
        "prompt": _WINDOW_SPEC,
        "cases": [
            _case([[1, 3, 2, 5, 4], 2], [3, 3, 5, 5], visible=True),
            _case([[1, 2], 5], [], visible=True),
            _case([[1, 2, 3], 0], [], label="k of zero is empty, not a crash"),
            _case([[1, 2, 3], -1], [], label="negative k is empty"),
            _case([[-5, -1, -9], 2], [-1, -1], label="negatives work normally"),
            _case([[4, 4, 4], 3], [4], label="a window covering everything gives one entry"),
        ],
    },
]

PROJECT_TASK_DESC = {
    "proj_calc": "Integer expression evaluator: precedence, truncation toward zero, unary minus.",
    "proj_report": "Summarise sales rows against a seventeen-point spec.",
    "proj_merge_gap": "Recursive config merge where the spec is deliberately silent on one case.",
    "proj_refactor_total": "Rewrite a working function against a changed requirement.",
    "proj_window_free": "Sliding-window maxima, in whichever language the model prefers.",
}

PROJECT_TASK_NOTES = {
    "proj_calc": "Three examples are given and five requirements are not demonstrated. "
                 "Truncation toward zero is the one models get wrong — Python's // floors, so "
                 "the obvious implementation returns -4 for -7/2 and passes every visible case.",
    "proj_report": "The long-spec case. Requirement 14 — underscore-prefixed products are "
                   "internal and excluded — sits two thirds down a seventeen-point list and is "
                   "shown by no example. A model that skims the opening and the examples cannot "
                   "pass it, which is the whole point of stating some tests and hiding others.",
    "proj_merge_gap": "The spec does not say what to do with two lists under one key. What is "
                      "graded is that the model chose a behaviour and did not crash on it — an "
                      "unspecified case is where real specs fail, and guessing safely is a "
                      "skill.",
    "proj_refactor_total": "Existing correct code plus a changed requirement, so the model must "
                           "read before writing. Half-up rounding is the trap: Python's round() "
                           "is banker's rounding and returns 50 where the spec wants 51.",
    "proj_window_free": "No language is named. The langpref suite shows models have strong "
                        "unprompted preferences; this asks whether the preference costs them "
                        "correctness when nothing forces the choice.",
}
