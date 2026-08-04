"""The benchmark suites: pure data, no logic.

Every task is a dict: id, prompt, entry, cases, optional lang (default python) and tier.
TASK_DESC carries the one-line "what this task actually tests" shown in reports and the API.

The stability contract: once a suite has shipped and runs have been recorded against it,
its tasks do not change — scores are only comparable against an identical suite. New
difficulty goes in a NEW suite (v1 -> v2 -> v3), never into an existing one.
"""

SUITES: dict[str, list[dict]] = {
    # coding-v2: modelled on HumanEval+ / BigCodeBench rather than on LeetCode.
    #
    # coding-v1 is saturated: seventeen of its eighteen tasks were solved perfectly in every one
    # of twenty-three runs, including its whole "hard" tier. Only `calculator` ever discriminated
    # — and the reason is the useful one. It is the single task where a *plausible* solution
    # fails: operator precedence, parentheses, unary minus. Everything else is recall of an
    # algorithm these models have memorised, so running them costs wall-clock and returns the
    # same answer for every model.
    #
    # So difficulty here is not exotic algorithms. It is specifications with traps and cases that
    # punish the obvious implementation: tie-breaking rules, escape sequences, validation that
    # must reject, output formats stated exactly. That is EvalPlus's finding too — the gap
    # between HumanEval and HumanEval+ is almost entirely adversarial inputs against code that
    # looked correct.
    "coding-v2": [
        {
            "id": "parse_query",
            "tier": "hard",
            "lang": "js",
            "prompt": (
                "Write a JavaScript function `parseQuery(qs)` that parses a URL query string "
                "into a plain object.\n"
                "Rules: a leading '?' is ignored. Pairs are separated by '&'; empty pairs are "
                "skipped. Each pair splits on the FIRST '=' only — later '=' characters belong "
                "to the value. A pair with no '=' maps the whole token to the empty string. "
                "'+' means a space, then percent-escapes are decoded. When a key repeats, its "
                "value becomes an array of all values in order. Return {} for an empty string."
                "\nReturn only the function in a single ```js code block."
            ),
            "entry": "parseQuery",
            "cases": [
                {"args": ["a=1&b=2"], "expect": {"a": "1", "b": "2"}},
                {"args": ["a=1&a=2&a=3"], "expect": {"a": ["1", "2", "3"]}},
                {"args": ["q=hello+world%21"], "expect": {"q": "hello world!"}},
                # No '=' at all: the token itself is the key.
                {"args": ["flag&x=1"], "expect": {"flag": "", "x": "1"}},
                {"args": ["?a=1"], "expect": {"a": "1"}},
                {"args": [""], "expect": {}},
                # Split on the FIRST '=': the value here is '%3D=' -> '=='.
                {"args": ["a=%3D=&b"], "expect": {"a": "==", "b": ""}},
            ],
        },
        {
            "id": "group_ranges",
            "tier": "core",
            "lang": "js",
            "prompt": (
                "Write a JavaScript function `groupRanges(nums)` that formats a strictly "
                "increasing array of integers as a compact range string.\n"
                "Consecutive runs collapse to 'start-end'; isolated values stand alone; parts "
                "join with ','. Two adjacent values are already a range ('1-2'). Negative "
                "numbers keep their sign, so -3..-1 renders as '-3--1'. An empty array "
                "returns ''.\nReturn only the function in a single ```js code block."
            ),
            "entry": "groupRanges",
            "cases": [
                {"args": [[1, 2, 3, 7, 9, 10, 11]], "expect": "1-3,7,9-11"},
                {"args": [[5]], "expect": "5"},
                {"args": [[]], "expect": ""},
                # The sign trap: a negative run still joins with '-'.
                {"args": [[-3, -2, -1, 0]], "expect": "-3-0"},
                {"args": [[1, 3, 5]], "expect": "1,3,5"},
                {"args": [[1, 2]], "expect": "1-2"},
            ],
        },
        {
            "id": "clamp_add",
            "tier": "core",
            "lang": "c",
            "prompt": (
                "Write a C function `int clamp_add(int a, int b)` returning a + b saturated "
                "to the int range: results above INT_MAX return INT_MAX, below INT_MIN return "
                "INT_MIN. The naive a + b overflows — do not rely on it. You may use any "
                "standard headers; the harness already includes stdio.h, stdlib.h, string.h "
                "and limits.h.\nReturn only the function in a single ```c code block."
            ),
            "entry": "clamp_add",
            "cases": [
                {"args": [100, 200], "expect": 300},
                {"args": [2147483647, 1], "expect": 2147483647},
                {"args": [-2147483648, -1], "expect": -2147483648},
                {"args": [2000000000, 2000000000], "expect": 2147483647},
                {"args": [-2000000000, -2000000000], "expect": -2147483648},
                {"args": [-2147483648, 2147483647], "expect": -1},
            ],
        },
        {
            "id": "round_to",
            "tier": "hard",
            "lang": "c",
            "prompt": (
                "Write a C function `int round_to(int n, int m)` rounding n to the nearest "
                "multiple of m (m > 0), with exact halves rounding AWAY from zero: 25 to the "
                "nearest 10 is 30, and -25 is -30. Beware: C integer division truncates "
                "toward zero, so the usual ((n + m/2) / m) * m is wrong for negative n.\n"
                "Return only the function in a single ```c code block."
            ),
            "entry": "round_to",
            "cases": [
                {"args": [25, 10], "expect": 30},
                {"args": [24, 10], "expect": 20},
                # The truncation trap.
                {"args": [-25, 10], "expect": -30},
                {"args": [-24, 10], "expect": -20},
                {"args": [7, 7], "expect": 7},
                {"args": [-3, 5], "expect": -5},
                {"args": [0, 3], "expect": 0},
            ],
        },
        {
            "id": "count_words",
            "tier": "core",
            "lang": "c",
            "prompt": (
                "Write a C function `int count_words(const char *s)` counting words in s, "
                "where a word is a maximal run of characters that are not spaces or tabs. "
                "Leading, trailing and repeated separators are all legal. Return 0 for an "
                "empty or all-separator string.\n"
                "Return only the function in a single ```c code block."
            ),
            "entry": "count_words",
            "cases": [
                {"args": ["hello world"], "expect": 2},
                {"args": ["  leading and trailing  "], "expect": 3},
                {"args": ["a\tb\t\tc"], "expect": 3},
                {"args": [""], "expect": 0},
                {"args": ["   "], "expect": 0},
                {"args": ["one"], "expect": 1},
            ],
        },
        {
            "id": "csv_escape",
            "tier": "core",
            "lang": "cpp",
            "prompt": (
                "Write a C++ function `std::string csv_escape(std::string field)` applying "
                "RFC 4180 quoting: if the field contains a comma or a double quote, wrap it "
                "in double quotes and double every interior quote; otherwise return it "
                "unchanged. An empty field stays empty and unquoted.\n"
                "Return only the function in a single ```cpp code block."
            ),
            "entry": "csv_escape",
            "cases": [
                {"args": ["plain"], "expect": "plain"},
                {"args": ["a,b"], "expect": "\"a,b\""},
                {"args": ["say \"hi\""], "expect": "\"say \"\"hi\"\"\""},
                {"args": [""], "expect": ""},
                {"args": ["both, \"q\""], "expect": "\"both, \"\"q\"\"\""},
                {"args": ["no quotes needed"], "expect": "no quotes needed"},
            ],
        },
        {
            "id": "balanced_depth",
            "tier": "hard",
            "lang": "cpp",
            "prompt": (
                "Write a C++ function `int balanced_depth(const std::string& s)` returning "
                "the maximum nesting depth of round brackets in s, or -1 if the brackets are "
                "not balanced (a closer without an opener, or unclosed openers at the end). "
                "Non-bracket characters are ignored. The empty string has depth 0.\n"
                "Return only the function in a single ```cpp code block."
            ),
            "entry": "balanced_depth",
            "cases": [
                {"args": ["(a(b)c)"], "expect": 2},
                {"args": ["()()()"], "expect": 1},
                {"args": [""], "expect": 0},
                {"args": [")("], "expect": -1},
                {"args": ["(()"], "expect": -1},
                {"args": ["x((y))z()"], "expect": 2},
            ],
        },
        {
            "id": "snake_to_camel",
            "tier": "core",
            "lang": "rust",
            "prompt": (
                "Write a Rust function `fn snake_to_camel(s: &str) -> String` converting "
                "snake_case to camelCase: every underscore is removed, and the character "
                "immediately after a run of underscores is uppercased if it is alphabetic "
                "(a digit stays as it is). Leading and trailing underscores vanish; a "
                "string with no underscores is unchanged.\n"
                "Return only the function in a single ```rust code block."
            ),
            "entry": "snake_to_camel",
            "cases": [
                {"args": ["hello_world"], "expect": "helloWorld"},
                {"args": ["already"], "expect": "already"},
                {"args": ["a__b"], "expect": "aB"},
                {"args": ["_leading"], "expect": "Leading"},
                {"args": ["trailing_"], "expect": "trailing"},
                {"args": ["x_1y"], "expect": "x1y"},
            ],
        },
        {
            "id": "mid_floor",
            "tier": "hard",
            "lang": "rust",
            "prompt": (
                "Write a Rust function `fn mid_floor(a: i64, b: i64) -> i64` returning the "
                "mathematical floor of (a + b) / 2. Two traps: a + b can overflow i64, and "
                "integer division truncates toward zero while floor rounds toward negative "
                "infinity — floor((-3 + 0) / 2) is -2, not -1.\n"
                "Return only the function in a single ```rust code block."
            ),
            "entry": "mid_floor",
            "cases": [
                {"args": [2, 6], "expect": 4},
                {"args": [-3, 0], "expect": -2},
                {"args": [9223372036854775807, 9223372036854775807],
                 "expect": 9223372036854775807},
                {"args": [-9223372036854775807, -9223372036854775807],
                 "expect": -9223372036854775807},
                {"args": [0, 0], "expect": 0},
                {"args": [-1, -2], "expect": -2},
            ],
        },
        {
            "id": "clamp_mul",
            "tier": "core",
            "lang": "csharp",
            "prompt": (
                "Write a C# static class `Sol` containing exactly one method, "
                "`public static int ClampMul(int a, int b)`, returning a * b saturated to the "
                "int range: results above int.MaxValue return int.MaxValue, below "
                "int.MinValue return int.MinValue. C# arithmetic is unchecked by default, so "
                "the naive a * b silently wraps.\n"
                "Return only the class in a single ```csharp code block."
            ),
            "entry": "Sol.ClampMul",
            "cases": [
                {"args": [7, 6], "expect": 42},
                {"args": [100000, 100000], "expect": 2147483647},
                {"args": [-100000, 100000], "expect": -2147483648},
                {"args": [46341, 46341], "expect": 2147483647},
                {"args": [-3, -4], "expect": 12},
                {"args": [0, 2147483647], "expect": 0},
            ],
        },
        {
            "id": "ordinal",
            "tier": "hard",
            "lang": "csharp",
            "prompt": (
                "Write a C# static class `Sol` containing exactly one method, "
                "`public static string Ordinal(int n)`, for n >= 0, appending the English "
                "ordinal suffix: 1st, 2nd, 3rd, 4th … The teens are the trap: 11th, 12th and "
                "13th take th despite ending in 1, 2, 3 — and so do 111, 112, 113.\n"
                "Return only the class in a single ```csharp code block."
            ),
            "entry": "Sol.Ordinal",
            "cases": [
                {"args": [1], "expect": "1st"},
                {"args": [2], "expect": "2nd"},
                {"args": [3], "expect": "3rd"},
                {"args": [4], "expect": "4th"},
                {"args": [11], "expect": "11th"},
                {"args": [12], "expect": "12th"},
                {"args": [13], "expect": "13th"},
                {"args": [21], "expect": "21st"},
                {"args": [111], "expect": "111th"},
                {"args": [102], "expect": "102nd"},
            ],
        },
        {
            "id": "slugify",
            "tier": "core",
            "lang": "php",
            "prompt": (
                "Write a PHP function `slugify(string $s): string` producing a URL slug: "
                "lowercase, every maximal run of non-alphanumeric characters becomes a single "
                "hyphen, and leading or trailing hyphens are trimmed. An input with nothing "
                "alphanumeric returns the empty string.\n"
                "Return only the function in a single ```php code block (no opening tag "
                "needed)."
            ),
            "entry": "slugify",
            "cases": [
                {"args": ["Hello, World!"], "expect": "hello-world"},
                {"args": ["  spaced   out  "], "expect": "spaced-out"},
                {"args": ["a--b__c"], "expect": "a-b-c"},
                {"args": ["!!!"], "expect": ""},
                {"args": ["Already-Fine"], "expect": "already-fine"},
                {"args": ["100% sure"], "expect": "100-sure"},
            ],
        },
        {
            "id": "pluck",
            "tier": "hard",
            "lang": "php",
            "prompt": (
                "Write a PHP function `pluck(array $rows, string $key): array` returning the "
                "value of $key from each associative row, in order, skipping rows where the "
                "key is ABSENT. The trap: a key present with a null value is kept as null — "
                "isset() cannot tell those apart, array_key_exists() can.\n"
                "Return only the function in a single ```php code block (no opening tag "
                "needed)."
            ),
            "entry": "pluck",
            "cases": [
                {"args": [[{"a": 1}, {"a": 2}], "a"], "expect": [1, 2]},
                {"args": [[{"a": 1}, {"b": 9}, {"a": 3}], "a"], "expect": [1, 3]},
                {"args": [[{"a": None}, {"a": 5}], "a"], "expect": [None, 5]},
                {"args": [[], "a"], "expect": []},
                {"args": [[{"x": "y"}], "a"], "expect": []},
            ],
        },
        {
            "id": "login_form",
            "tier": "core",
            "lang": "html",
            "prompt": (
                "Write an HTML fragment (no document shell): a form that POSTs to /login "
                "containing an email input and a password input, each with its own <label> "
                "bound via for/id, both marked required, and a <button> reading exactly "
                "'Sign in'.\nReturn only the fragment in a single ```html code block."
            ),
            "entry": "login_form",
            "cases": [
                {"op": "count", "sel": "form", "expect": 1},
                {"op": "attr", "sel": "form", "name": "method", "expect": "post"},
                {"op": "attr", "sel": "form", "name": "action", "expect": "/login"},
                {"op": "count", "sel": "input[type=email][required]", "expect": 1},
                {"op": "count", "sel": "input[type=password][required]", "expect": 1},
                {"op": "labels_bound", "expect": 2},
                {"op": "text", "sel": "button", "expect": "Sign in"},
            ],
        },
        {
            "id": "data_table",
            "tier": "hard",
            "lang": "html",
            "prompt": (
                "Write an HTML fragment: a table of quarterly revenue with a <caption> "
                "reading exactly 'Quarterly revenue', a <thead> whose single row holds three "
                "<th> cells each with scope=\"col\", and a <tbody> of exactly two rows of "
                "three <td> cells. Cell contents are yours to choose.\n"
                "Return only the fragment in a single ```html code block."
            ),
            "entry": "data_table",
            "cases": [
                {"op": "count", "sel": "table", "expect": 1},
                {"op": "text", "sel": "caption", "expect": "Quarterly revenue"},
                {"op": "count", "sel": "thead th", "expect": 3},
                {"op": "count", "sel": "thead th[scope=col]", "expect": 3},
                {"op": "count", "sel": "tbody tr", "expect": 2},
                {"op": "count", "sel": "tbody td", "expect": 6},
            ],
        },
        {
            "id": "card_grid",
            "tier": "core",
            "lang": "css",
            "prompt": (
                "Write CSS: a .cards container laid out as a grid with columns "
                "repeat(auto-fill, minmax(240px, 1fr)) and a 16px gap; and a .card rule with "
                "border-radius 8px whose :hover state sets transform to translateY(-2px).\n"
                "Return only the CSS in a single ```css code block."
            ),
            "entry": "card_grid",
            "cases": [
                {"op": "decl", "sel": ".cards", "prop": "display", "expect": "grid"},
                {"op": "decl", "sel": ".cards", "prop": "grid-template-columns",
                 "expect": "repeat(auto-fill, minmax(240px, 1fr))"},
                {"op": "decl", "sel": ".cards", "prop": "gap", "expect": "16px"},
                {"op": "decl", "sel": ".card", "prop": "border-radius", "expect": "8px"},
                {"op": "decl", "sel": ".card:hover", "prop": "transform",
                 "expect": "translateY(-2px)"},
            ],
        },
        {
            "id": "theme_vars",
            "tier": "hard",
            "lang": "css",
            "prompt": (
                "Write CSS implementing a two-theme token: :root defines --ink as #111111; "
                "inside an @media (prefers-color-scheme: dark) block, :root redefines --ink "
                "as #eeeeee; and body sets color to var(--ink). The trap is scope — the dark "
                "value must live inside the media query, not beside it.\n"
                "Return only the CSS in a single ```css code block."
            ),
            "entry": "theme_vars",
            "cases": [
                {"op": "decl", "sel": ":root", "prop": "--ink", "expect": "#111111"},
                {"op": "decl", "sel": ":root", "prop": "--ink", "expect": "#eeeeee",
                 "media": "prefers-color-scheme: dark"},
                {"op": "decl", "sel": "body", "prop": "color", "expect": "var(--ink)"},
            ],
        },
        {
            "id": "semver_cmp",
            "tier": "core",
            "prompt": (
                "Write a Python function `semver_cmp(a: str, b: str) -> int` comparing two "
                "semantic versions, returning -1 if a < b, 0 if equal, 1 if a > b.\n"
                "Rules: compare major, minor, patch numerically. A version WITH a pre-release "
                "tag has LOWER precedence than the same version without one (1.0.0-alpha < "
                "1.0.0). Compare pre-release by splitting on '.': identifiers that are all "
                "digits compare numerically and rank lower than non-numeric identifiers, which "
                "compare as ASCII strings. A longer pre-release with an otherwise equal prefix "
                "is greater. Build metadata after '+' is ignored entirely.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "semver_cmp",
            "cases": [
                {"args": ["1.0.0", "1.0.1"], "expect": -1},
                {"args": ["1.2.0", "1.10.0"], "expect": -1},
                # The trap: a pre-release is LOWER than the release it precedes.
                {"args": ["1.0.0-alpha", "1.0.0"], "expect": -1},
                {"args": ["1.0.0-alpha", "1.0.0-alpha.1"], "expect": -1},
                {"args": ["1.0.0-alpha.1", "1.0.0-alpha.beta"], "expect": -1},
                {"args": ["1.0.0-beta.2", "1.0.0-beta.11"], "expect": -1},
                {"args": ["1.0.0+build.9", "1.0.0+build.1"], "expect": 0},
                {"args": ["2.0.0", "2.0.0"], "expect": 0},
            ],
        },
        {
            "id": "csv_line",
            "tier": "core",
            "prompt": (
                "Write a Python function `csv_line(line: str) -> list[str]` splitting one CSV "
                "record into fields.\n"
                "Rules: fields are comma separated. A field may be wrapped in double quotes, in "
                "which case it may contain commas and doubled quotes ("
                '"" ) which represent one literal quote character. Quotes only have meaning at '
                "the very start of a field. Do not strip whitespace. An empty input returns "
                "['']. Do not use the csv module.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "csv_line",
            "cases": [
                {"args": ["a,b,c"], "expect": ["a", "b", "c"]},
                {"args": ['a,"b,c",d'], "expect": ["a", "b,c", "d"]},
                {"args": ['"he said ""hi""",x'], "expect": ['he said "hi"', "x"]},
                {"args": ["a,,c"], "expect": ["a", "", "c"]},
                {"args": [""], "expect": [""]},
                # Quote mid-field is literal, not a delimiter.
                {"args": ['a,b"c,d'], "expect": ["a", 'b"c', "d"]},
                {"args": [' a , b '], "expect": [" a ", " b "]},
                {"args": ['""'], "expect": [""]},
            ],
        },
        {
            "id": "glob_match",
            "tier": "hard",
            "prompt": (
                "Write a Python function `glob_match(pattern: str, text: str) -> bool` returning "
                "whether pattern matches the whole of text. '?' matches exactly one character. "
                "'*' matches any sequence including empty. All other characters match "
                "themselves. Do not use the re, fnmatch or glob modules.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "glob_match",
            "cases": [
                {"args": ["a*c", "abbbc"], "expect": True},
                {"args": ["a?c", "abc"], "expect": True},
                {"args": ["a?c", "ac"], "expect": False},
                # Greedy matching fails this one.
                {"args": ["*a*b", "aaab"], "expect": True},
                {"args": ["*", ""], "expect": True},
                {"args": ["", ""], "expect": True},
                {"args": ["", "a"], "expect": False},
                {"args": ["**a", "a"], "expect": True},
                {"args": ["a*", "a"], "expect": True},
                {"args": ["*b*", "abc"], "expect": True},
            ],
        },
        {
            "id": "roman_strict",
            "tier": "hard",
            "prompt": (
                "Write a Python function `roman_strict(s: str) -> int | None` converting a "
                "canonical uppercase Roman numeral (1-3999) to an int, returning None if s is "
                "not canonical.\n"
                "Canonical means: I, X, C repeat at most three times; V, L, D never repeat; the "
                "only subtractive pairs are IV, IX, XL, XC, CD, CM and each may appear at most "
                "once; symbols otherwise appear in non-increasing value order. 'IIII', 'VV', "
                "'IC' and 'MCMC' are all invalid.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "roman_strict",
            "cases": [
                {"args": ["MCMXCIV"], "expect": 1994},
                {"args": ["III"], "expect": 3},
                {"args": ["IV"], "expect": 4},
                {"args": ["MMMCMXCIX"], "expect": 3999},
                # Everything below must be rejected; a permissive parser returns a number.
                {"args": ["IIII"], "expect": None},
                {"args": ["VV"], "expect": None},
                {"args": ["IC"], "expect": None},
                {"args": ["MCMC"], "expect": None},
                {"args": [""], "expect": None},
                {"args": ["XLIX"], "expect": 49},
            ],
        },
        {
            "id": "topo_lex",
            "tier": "hard",
            "prompt": (
                "Write a Python function `topo_lex(n: int, edges: list) -> list | None` returning "
                "a topological ordering of nodes 0..n-1 given a list of [u, v] edges meaning u "
                "must come before v. Among all valid orderings return the lexicographically "
                "smallest. Return None if the graph has a cycle. Duplicate edges may appear.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "topo_lex",
            "cases": [
                {"args": [4, [[0, 1], [1, 2], [2, 3]]], "expect": [0, 1, 2, 3]},
                # A plain DFS or arbitrary-queue topo sort gets a valid but non-minimal order.
                {"args": [4, [[2, 3]]], "expect": [0, 1, 2, 3]},
                {"args": [3, [[1, 0]]], "expect": [1, 0, 2]},
                {"args": [2, [[0, 1], [1, 0]]], "expect": None},
                {"args": [1, []], "expect": [0]},
                {"args": [3, [[0, 2], [0, 2]]], "expect": [0, 1, 2]},
                {"args": [5, [[4, 0], [4, 1]]], "expect": [2, 3, 4, 0, 1]},
            ],
        },
        {
            "id": "lru_ops",
            "tier": "core",
            "prompt": (
                "Write a Python function `lru_ops(capacity: int, ops: list) -> list` simulating "
                "an LRU cache. Each op is ['put', key, value] or ['get', key]. Return the list "
                "of results of the 'get' operations only, using -1 for a miss.\n"
                "A get counts as a use. A put of an existing key updates its value and counts as "
                "a use. When capacity is exceeded evict the least recently used key.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "lru_ops",
            "cases": [
                {"args": [2, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3],
                              ["get", 2], ["get", 3]]], "expect": [1, -1, 3]},
                # put on an existing key must refresh recency, not just overwrite.
                {"args": [2, [["put", 1, 1], ["put", 2, 2], ["put", 1, 10], ["put", 3, 3],
                              ["get", 2], ["get", 1]]], "expect": [-1, 10]},
                {"args": [1, [["put", 1, 1], ["put", 2, 2], ["get", 1]]], "expect": [-1]},
                {"args": [2, [["get", 9]]], "expect": [-1]},
                {"args": [3, [["put", 1, 1], ["put", 2, 2], ["put", 3, 3], ["get", 1],
                              ["put", 4, 4], ["get", 2], ["get", 1]]], "expect": [1, -1, 1]},
            ],
        },
        {
            "id": "justify",
            "tier": "hard",
            "prompt": (
                "Write a Python function `justify(words: list, width: int) -> list` performing "
                "full text justification. Pack as many words per line as fit with at least one "
                "space between them. Pad each line to exactly width by distributing spaces "
                "between words as evenly as possible, putting the extra spaces on the LEFT gaps "
                "first. The last line, and any line holding a single word, is left justified "
                "with the remaining space on the right.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "justify",
            "cases": [
                {"args": [["This", "is", "an", "example", "of", "text", "justification."], 16],
                 "expect": ["This    is    an", "example  of text", "justification.  "]},
                {"args": [["What", "must", "be", "acknowledgment", "shall", "be"], 16],
                 "expect": ["What   must   be", "acknowledgment  ", "shall be        "]},
                {"args": [["a"], 3], "expect": ["a  "]},
                {"args": [["aa", "bb"], 5], "expect": ["aa bb"]},
            ],
        },
        {
            "id": "path_norm",
            "tier": "core",
            "prompt": (
                "Write a Python function `path_norm(path: str) -> str` normalising an absolute "
                "POSIX path. Collapse repeated slashes, resolve '.' and '..', and remove any "
                "trailing slash. '..' at the root stays at the root. The result always starts "
                "with '/' and is '/' when nothing remains. Do not use os.path or pathlib.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "path_norm",
            "cases": [
                {"args": ["/home/"], "expect": "/home"},
                {"args": ["/a/./b/../../c/"], "expect": "/c"},
                # Escaping above root is the trap.
                {"args": ["/../"], "expect": "/"},
                {"args": ["/a//b////c/d//././/.."], "expect": "/a/b/c"},
                {"args": ["/"], "expect": "/"},
                {"args": ["/..hidden"], "expect": "/..hidden"},
                {"args": ["/a/../../b"], "expect": "/b"},
            ],
        },
        {
            "id": "json_pointer",
            "tier": "hard",
            "prompt": (
                "Write a Python function `json_pointer(doc, pointer: str)` resolving an RFC 6901 "
                "JSON Pointer against doc, returning the referenced value or None if it does not "
                "resolve.\n"
                "The empty pointer '' returns doc itself. Otherwise the pointer starts with '/' "
                "and is split on '/' into reference tokens. In each token '~1' decodes to '/' and "
                "'~0' decodes to '~', and '~1' must be decoded BEFORE '~0'. A token addressing a "
                "list must be all digits with no leading zeros (except '0' itself) and in range. "
                "Return None rather than raising.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "json_pointer",
            "cases": [
                {"args": [{"a": {"b": [1, 2, 3]}}, "/a/b/1"], "expect": 2},
                {"args": [{"a": 1}, ""], "expect": {"a": 1}},
                {"args": [{"a/b": 7}, "/a~1b"], "expect": 7},
                {"args": [{"m~n": 8}, "/m~0n"], "expect": 8},
                # Decoding ~0 first would turn ~01 into / and find the wrong key.
                {"args": [{"~1": 9}, "/~01"], "expect": 9},
                {"args": [{"a": [1]}, "/a/01"], "expect": None},
                {"args": [{"a": [1]}, "/a/5"], "expect": None},
                {"args": [{"a": 1}, "/b"], "expect": None},
            ],
        },
        {
            "id": "base_convert",
            "tier": "core",
            "prompt": (
                "Write a Python function `base_convert(s: str, frm: int, to: int) -> str | None` "
                "converting the integer written in base `frm` to base `to`, for bases 2..36.\n"
                "Input may have a leading '-' and leading zeros, and uses digits 0-9 then "
                "letters case-insensitively. Output uses lowercase letters, has no leading zeros, "
                "and zero is '0' with no sign. Return None if s is empty, is only a sign, "
                "contains a digit not valid in base frm, or if either base is out of range.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "base_convert",
            "cases": [
                {"args": ["ff", 16, 2], "expect": "11111111"},
                {"args": ["-1A", 16, 10], "expect": "-26"},
                {"args": ["0007", 10, 10], "expect": "7"},
                {"args": ["0", 10, 36], "expect": "0"},
                {"args": ["-0", 10, 2], "expect": "0"},
                {"args": ["z", 36, 10], "expect": "35"},
                {"args": ["2", 2, 10], "expect": None},
                {"args": ["", 10, 2], "expect": None},
                {"args": ["-", 10, 2], "expect": None},
                {"args": ["10", 1, 10], "expect": None},
            ],
        },
        {
            "id": "interval_intersect",
            "tier": "core",
            "prompt": (
                "Write a Python function `interval_intersect(a: list, b: list) -> list` returning "
                "the intersection of two lists of closed intervals [start, end]. Each input list "
                "is sorted and its own intervals are disjoint, but intervals that merely touch "
                "(one ends where the next begins) DO intersect and the shared point is a valid "
                "result interval. Return the intersections in order.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "interval_intersect",
            "cases": [
                {"args": [[[0, 2], [5, 10]], [[1, 5], [8, 12]]],
                 "expect": [[1, 2], [5, 5], [8, 10]]},
                {"args": [[], [[1, 2]]], "expect": []},
                {"args": [[[1, 3]], [[3, 5]]], "expect": [[3, 3]]},
                {"args": [[[1, 10]], [[2, 3], [4, 5]]], "expect": [[2, 3], [4, 5]]},
                {"args": [[[1, 2]], [[3, 4]]], "expect": []},
            ],
        },
        {
            "id": "tokenize_expr",
            "tier": "hard",
            "prompt": (
                "Write a Python function `tokenize_expr(s: str) -> list | None` tokenising an "
                "arithmetic expression into a list of tokens.\n"
                "Numbers are non-negative integers or decimals like '3' or '2.5' and become "
                "floats if they contain '.', ints otherwise. Operators are '+', '-', '*', '/' "
                "and parentheses '(' and ')', each its own single-character string token. "
                "Whitespace separates tokens but is not a token. A '-' is always an operator "
                "token, never part of a number. Return None if any other character appears or if "
                "a number has more than one '.'.\n"
                "Return only the function in a single ```python code block."
            ),
            "entry": "tokenize_expr",
            "cases": [
                {"args": ["1+2"], "expect": [1, "+", 2]},
                {"args": ["  12 * ( 3.5 - 4 ) "], "expect": [12, "*", "(", 3.5, "-", 4, ")"]},
                {"args": ["-3"], "expect": ["-", 3]},
                {"args": ["2--3"], "expect": [2, "-", "-", 3]},
                {"args": ["10.25/2"], "expect": [10.25, "/", 2]},
                {"args": ["1.2.3"], "expect": None},
                {"args": ["1$2"], "expect": None},
                {"args": [""], "expect": []},
            ],
        },
    ],
    "coding-v1": [
        {
            "id": "binary_search",
            "prompt": ("Write a Python function `binary_search(items: list[int], target: int) -> int` "
                       "that returns the index of target in the sorted list items, or -1 if absent. "
                       "Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "binary_search",
            "cases": [
                {"args": [[1, 3, 5, 7, 9], 7], "expect": 3},
                {"args": [[1, 3, 5, 7, 9], 1], "expect": 0},
                {"args": [[1, 3, 5, 7, 9], 9], "expect": 4},
                {"args": [[1, 3, 5, 7, 9], 4], "expect": -1},
                {"args": [[], 1], "expect": -1},
                {"args": [[42], 42], "expect": 0},
            ],
        },
        {
            "id": "merge_intervals",
            "prompt": ("Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                       "that merges all overlapping intervals and returns them sorted by start. "
                       "Touching intervals like [1,2] and [2,3] count as overlapping. "
                       "Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "merge_intervals",
            "cases": [
                {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expect": [[1, 6], [8, 10], [15, 18]]},
                {"args": [[[1, 4], [4, 5]]], "expect": [[1, 5]]},
                {"args": [[]], "expect": []},
                {"args": [[[5, 6], [1, 2]]], "expect": [[1, 2], [5, 6]]},
            ],
        },
        {
            "id": "word_freq",
            "prompt": ("Write a Python function `word_freq(text: str, n: int) -> list[tuple[str, int]]` "
                       "returning the n most common lowercase words in text, as (word, count) tuples "
                       "sorted by count descending then word ascending. Words are runs of letters; "
                       "ignore case and punctuation. Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "word_freq",
            "cases": [
                {"args": ["the cat the dog THE bird", 2], "expect": [["the", 3], ["bird", 1]]},
                {"args": ["a b c", 2], "expect": [["a", 1], ["b", 1]]},
                {"args": ["", 3], "expect": []},
            ],
        },
        {
            "id": "roman",
            "prompt": ("Write a Python function `to_roman(n: int) -> str` converting an integer "
                       "between 1 and 3999 into a Roman numeral. Return only the function in a "
                       "single ```python code block."),
            "tier": "core",
            "entry": "to_roman",
            "cases": [
                {"args": [1], "expect": "I"},
                {"args": [4], "expect": "IV"},
                {"args": [9], "expect": "IX"},
                {"args": [58], "expect": "LVIII"},
                {"args": [1994], "expect": "MCMXCIV"},
                {"args": [3999], "expect": "MMMCMXCIX"},
            ],
        },
        {
            "id": "balanced",
            "prompt": ("Write a Python function `is_balanced(s: str) -> bool` returning True when "
                       "every (), [] and {} in s is correctly matched and nested. Ignore all other "
                       "characters. Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "is_balanced",
            "cases": [
                {"args": ["({[]})"], "expect": True},
                {"args": ["(]"], "expect": False},
                {"args": [""], "expect": True},
                {"args": ["a(b)c[d]{e}"], "expect": True},
                {"args": ["((("], "expect": False},
            ],
        },
        {
            "id": "flatten",
            "prompt": ("Write a Python function `flatten(nested: list) -> list` that fully flattens "
                       "an arbitrarily nested list of lists into a single flat list, preserving order. "
                       "Non-list values pass through. Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "flatten",
            "cases": [
                {"args": [[1, [2, [3, [4]]], 5]], "expect": [1, 2, 3, 4, 5]},
                {"args": [[]], "expect": []},
                {"args": [[[], [[]], [[[1]]]]], "expect": [1]},
            ],
        },
        {
            "id": "two_sum",
            "prompt": ("Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` "
                       "returning the indices of the two numbers that add to target, as a list of two "
                       "ascending indices. Exactly one solution exists and an element can't be reused. "
                       "Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "two_sum",
            "cases": [
                {"args": [[2, 7, 11, 15], 9], "expect": [0, 1]},
                {"args": [[3, 2, 4], 6], "expect": [1, 2]},
                {"args": [[3, 3], 6], "expect": [0, 1]},
                {"args": [[-1, -2, -3, -4], -7], "expect": [2, 3]},
            ],
        },
        {
            "id": "lru_cache_sim",
            "prompt": ("Write a Python function `lru(capacity: int, ops: list) -> list` simulating an "
                       "LRU cache. ops is a list of ['put', key, value] or ['get', key]. Return the list "
                       "of results of every 'get', using -1 for a miss. A get or put counts as a use. "
                       "Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "lru",
            "cases": [
                {"args": [2, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["get", 3]]],
                 "expect": [1, -1, 3]},
                {"args": [1, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["get", 2]]], "expect": [-1, 2]},
            ],
        },
        {
            "id": "group_anagrams",
            "prompt": ("Write a Python function `group_anagrams(words: list[str]) -> list[list[str]]` "
                       "grouping words that are anagrams. Each group is sorted alphabetically, and the "
                       "groups are sorted by their first word. Return only the function in a single "
                       "```python code block."),
            "tier": "core",
            "entry": "group_anagrams",
            "cases": [
                {"args": [["eat", "tea", "tan", "ate", "nat", "bat"]],
                 "expect": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]},
                {"args": [[]], "expect": []},
                {"args": [["a"]], "expect": [["a"]]},
            ],
        },
        {
            "id": "run_length",
            "prompt": ("Write a Python function `rle(s: str) -> str` performing run-length encoding: "
                       "each run becomes the character followed by its count, including runs of length 1 "
                       "(so 'aab' becomes 'a2b1'). Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "rle",
            "cases": [
                {"args": ["aab"], "expect": "a2b1"},
                {"args": [""], "expect": ""},
                {"args": ["abc"], "expect": "a1b1c1"},
                {"args": ["aaaaaaaaaaaa"], "expect": "a12"},
            ],
        },
        {
            "id": "compare_versions",
            "prompt": ("Write a Python function `compare_versions(a: str, b: str) -> int` comparing "
                       "dotted version strings. Return -1 if a < b, 1 if a > b, 0 if equal. Segments "
                       "are integers, missing segments count as 0, and leading zeros are ignored "
                       "('1.02' equals '1.2'). Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "compare_versions",
            "cases": [
                {"args": ["1.2", "1.10"], "expect": -1},
                {"args": ["1.02", "1.2"], "expect": 0},
                {"args": ["1.0.0", "1"], "expect": 0},
                {"args": ["2.1", "2.0.9"], "expect": 1},
                {"args": ["1.0.1", "1"], "expect": 1},
            ],
        },
        {
            "id": "spiral",
            "prompt": ("Write a Python function `spiral(matrix: list[list[int]]) -> list[int]` returning "
                       "the elements of the matrix in clockwise spiral order starting at the top-left. "
                       "Return only the function in a single ```python code block."),
            "tier": "core",
            "entry": "spiral",
            "cases": [
                {"args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], "expect": [1, 2, 3, 6, 9, 8, 7, 4, 5]},
                {"args": [[[1, 2], [3, 4]]], "expect": [1, 2, 4, 3]},
                {"args": [[]], "expect": []},
                {"args": [[[1]]], "expect": [1]},
            ],
        },
        # ---- hard tier ---------------------------------------------------------------------
        # The core tier is a smoke test: any usable coding model clears it, which means it can
        # confirm a model works but cannot rank two that both do. These are chosen to fail
        # partially — each has an edge case that a plausible-looking implementation gets wrong.
        {
            "id": "edit_distance",
            "tier": "hard",
            "prompt": ("Write a Python function `edit_distance(a: str, b: str) -> int` returning the "
                       "Levenshtein distance between a and b: the minimum number of single-character "
                       "insertions, deletions or substitutions to turn a into b. "
                       "Return only the function in a single ```python code block."),
            "entry": "edit_distance",
            "cases": [
                {"args": ["kitten", "sitting"], "expect": 3},
                {"args": ["flaw", "lawn"], "expect": 2},
                {"args": ["", "abc"], "expect": 3},
                {"args": ["same", "same"], "expect": 0},
                {"args": ["a", ""], "expect": 1},
                {"args": ["intention", "execution"], "expect": 5},
            ],
        },
        {
            "id": "lis_length",
            "tier": "hard",
            "prompt": ("Write a Python function `lis_length(nums: list[int]) -> int` returning the "
                       "length of the longest STRICTLY increasing subsequence (elements need not be "
                       "contiguous). Return only the function in a single ```python code block."),
            "entry": "lis_length",
            "cases": [
                {"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expect": 4},
                {"args": [[0, 1, 0, 3, 2, 3]], "expect": 4},
                {"args": [[7, 7, 7, 7]], "expect": 1},
                {"args": [[]], "expect": 0},
                {"args": [[3, 1, 2]], "expect": 2},
            ],
        },
        {
            "id": "simplify_path",
            "tier": "hard",
            "prompt": ("Write a Python function `simplify_path(path: str) -> str` that canonicalises "
                       "an absolute Unix path: collapse repeated slashes, resolve '.' and '..', and "
                       "return a path with no trailing slash (except the root, which is '/'). "
                       "'..' at the root stays at the root. "
                       "Return only the function in a single ```python code block."),
            "entry": "simplify_path",
            "cases": [
                {"args": ["/home/"], "expect": "/home"},
                {"args": ["/../"], "expect": "/"},
                {"args": ["/home//foo/"], "expect": "/home/foo"},
                {"args": ["/a/./b/../../c/"], "expect": "/c"},
                {"args": ["/"], "expect": "/"},
                {"args": ["/a/../../b/../c//.//"], "expect": "/c"},
            ],
        },
        {
            "id": "calculator",
            "tier": "hard",
            "prompt": ("Write a Python function `calculate(expr: str) -> int` evaluating an arithmetic "
                       "expression containing non-negative integers and the operators + - * / with "
                       "normal precedence and optional spaces. There are no parentheses. Division is "
                       "INTEGER division that truncates toward zero (so 7/-2 would be -3, and 3/2 is 1). "
                       "Return only the function in a single ```python code block."),
            "entry": "calculate",
            "cases": [
                {"args": ["3+2*2"], "expect": 7},
                {"args": [" 3/2 "], "expect": 1},
                {"args": [" 3+5 / 2 "], "expect": 5},
                {"args": ["2*3+4*5"], "expect": 26},
                {"args": ["14-3/2"], "expect": 13},
                {"args": ["100"], "expect": 100},
            ],
        },
        {
            "id": "word_break",
            "tier": "hard",
            "prompt": ("Write a Python function `word_break(s: str, words: list[str]) -> bool` "
                       "returning True when s can be segmented into a sequence of one or more words "
                       "from the list. Words may be reused. "
                       "Return only the function in a single ```python code block."),
            "entry": "word_break",
            "cases": [
                {"args": ["leetcode", ["leet", "code"]], "expect": True},
                {"args": ["applepenapple", ["apple", "pen"]], "expect": True},
                {"args": ["catsandog", ["cats", "dog", "sand", "and", "cat"]], "expect": False},
                {"args": ["aaaaaaa", ["aaaa", "aaa"]], "expect": True},
                {"args": ["", ["a"]], "expect": True},
            ],
        },
        {
            "id": "topo_sort",
            "tier": "hard",
            "prompt": ("Write a Python function `topo_sort(n: int, edges: list[list[int]]) -> list[int]` "
                       "returning a topological ordering of nodes 0..n-1, where each edge [a, b] means a "
                       "must come before b. When several orderings are valid, return the "
                       "lexicographically smallest. Return an empty list if a cycle makes it impossible. "
                       "Return only the function in a single ```python code block."),
            "entry": "topo_sort",
            "cases": [
                {"args": [4, [[0, 1], [0, 2], [1, 3], [2, 3]]], "expect": [0, 1, 2, 3]},
                {"args": [3, [[2, 0], [2, 1]]], "expect": [2, 0, 1]},
                {"args": [2, [[0, 1], [1, 0]]], "expect": []},
                {"args": [3, []], "expect": [0, 1, 2]},
            ],
        },
    ],
}


TASK_DESC = {
    "binary_search": "Find a value's index in a sorted list",
    "merge_intervals": "Merge overlapping intervals",
    "word_freq": "Top-n most common words with tie rules",
    "roman": "Integer to Roman numeral",
    "balanced": "Bracket matching for () [] {}",
    "flatten": "Flatten arbitrarily nested lists",
    "two_sum": "Indices of the pair summing to a target",
    "lru_cache_sim": "Simulate an LRU cache's get/put sequence",
    "group_anagrams": "Group words that are anagrams",
    "run_length": "Run-length encode a string",
    "compare_versions": "Compare dotted version strings",
    "spiral": "Matrix in clockwise spiral order",
    "edit_distance": "Levenshtein distance",
    "lis_length": "Longest strictly increasing subsequence",
    "simplify_path": "Canonicalise a Unix path with . and ..",
    "calculator": "Arithmetic with precedence, no eval",
    "word_break": "Segment a string into dictionary words",
    "topo_sort": "Topological sort of a dependency graph",
    "parse_query": "Parse a URL query string into an object",
    "group_ranges": "Collapse consecutive integers into range strings",
    "clamp_add": "Saturating int addition — overflow trap",
    "round_to": "Round to nearest multiple, halves away from zero",
    "count_words": "Count words split on spaces and tabs",
    "csv_escape": "RFC 4180 CSV field quoting",
    "balanced_depth": "Max bracket nesting depth, -1 if unbalanced",
    "snake_to_camel": "snake_case to camelCase — digits stop capitalisation",
    "mid_floor": "Floor midpoint of two i64s — overflow and negatives",
    "clamp_mul": "Saturating int multiplication",
    "ordinal": "English ordinal suffix — the 11th/12th/13th trap",
    "slugify": "URL slug: symbol runs become one hyphen",
    "pluck": "Column from associative rows — null vs missing key",
    "login_form": "Login form with labels bound to their inputs",
    "data_table": "Revenue table with caption and scoped headers",
    "card_grid": "Responsive auto-fill card grid",
    "theme_vars": "Dark-mode token inside a media query",
    "semver_cmp": "Semantic versions incl. pre-release precedence",
    "csv_line": "Split one CSV record honouring quotes",
    "glob_match": "Glob matching with ? and *",
    "roman_strict": "Roman to int, rejecting non-canonical forms",
    "topo_lex": "Smallest topological order, None on cycle",
    "lru_ops": "LRU cache with eviction order",
    "justify": "Full text justification",
    "path_norm": "Normalise a POSIX path with . and ..",
    "json_pointer": "Resolve an RFC 6901 JSON Pointer",
    "base_convert": "Integer between bases 2-36 with validation",
    "interval_intersect": "Intersect two interval lists",
    "tokenize_expr": "Tokenise arithmetic, None on invalid input",
}

