"""langpref-v1: which language does a model REACH FOR when you don't tell it?

Every other suite here asks whether the answer works. This one asks what the answer is
written in. Each task is a real problem with a pull toward some language — call a C++
library, automate Windows, blink an LED, build a website — and is deliberately silent about
which language to use. What comes back reveals a disposition: a model that answers every
prompt in Python is a different colleague from one that reaches for C# on Windows and Go for
throughput, and neither shows up in a correctness score.

The score is secondary and exists only so an outright strange choice is visible — a browser
DOM answer written in C++ fails, while any defensible pick passes. The real output is the
PROFILE: what each model chose, task by task, rendered as a distribution in the report.
Nothing here is graded on whether the code runs, so a task is never wrong for being
ambitious.

Prompt discipline matters more than usual: no task may name a language, hint at a file
extension, or mention a library that only exists in one ecosystem, because any of those
would be measuring the prompt rather than the model.
"""


def _t(tid, tier, prompt, defensible, desc, note):
    return {
        "id": tid, "category": "preference", "tier": tier, "lang": "langpick",
        "entry": "answer", "prompt": prompt,
        "cases": [{"check": "language", "expect_any": defensible,
                   "label": "picked a language that fits the problem"}],
        "_desc": desc, "_note": note,
    }


LANGPREF_TASKS = [
    _t("pref_native_api", "hard",
       "A shared library on disk exposes this C ABI:\n\n"
       "    double distance(double x1, double y1, double x2, double y2);\n"
       "    int    path_length(const char* map, int width, int height);\n\n"
       "Write a program that loads that library, calls both functions with sample "
       "inputs, and prints the results. Give me the complete implementation.",
       ["cpp", "c", "python", "rust", "go", "csharp", "javascript"],
       "Call a C ABI: native, or bind it from a scripting language?",
       "The pull is toward C/C++ — it is the library's own language and needs no binding "
       "layer. Reaching for Python ctypes or cffi is equally defensible and much more "
       "common in practice; which one a model picks says whether it thinks in the "
       "problem's ecosystem or in its default one."),

    _t("pref_website", "core",
       "Build a small internal website that lists products from a database and lets "
       "someone search them by name. Include the server side and the page it serves.",
       ["python", "javascript", "typescript", "csharp", "php", "go", "ruby", "java"],
       "A database-backed website with search",
       "The widest-open task in the suite: Flask, Express, ASP.NET, Laravel and Rails are "
       "all ordinary answers. Whatever comes back is close to the model's true default for "
       "'write me a web app', which is the single most requested thing anyone asks a "
       "coding model for."),

    _t("pref_windows_admin", "hard",
       "On a Windows server, produce something that finds every process using more than "
       "500 MB of memory, writes them to a timestamped CSV, and can run every 5 minutes "
       "unattended.",
       ["powershell", "python", "csharp", "batch"],
       "Windows ops automation on a schedule",
       "PowerShell is the native answer — Get-Process, Export-Csv and a scheduled task, no "
       "dependencies on a locked-down box. A model that answers in Python here is telling "
       "you it optimises for its own fluency over the platform's grain; on Windows that is "
       "an install request and an approval ticket."),

    _t("pref_throughput_service", "hard",
       "Write a service that accepts a very high rate of small webhook POSTs — tens of "
       "thousands per second — validates a signature header, and pushes each body onto a "
       "queue. Latency matters more than features.",
       ["go", "rust", "java", "cpp", "csharp", "javascript", "typescript", "python"],
       "A latency-sensitive high-throughput ingest service",
       "The stated constraints point at Go or Rust. Python here is not indefensible with "
       "the right stack, but choosing it while the prompt says 'tens of thousands per "
       "second' shows the model weighting familiarity over the requirement it was just "
       "given."),

    _t("pref_embedded_blink", "core",
       "On an ATmega328 microcontroller with an LED wired to digital pin 13, make the LED "
       "blink with a 500 ms period. Give the full program.",
       ["cpp", "c", "rust", "assembly"],
       "Bare-metal microcontroller blink",
       "The narrowest task here: an 8-bit AVR runs C or C++ (an Arduino sketch is C++). A "
       "Python answer would be a category error, so this doubles as a check that the "
       "detector and the model are both grounded."),

    _t("pref_data_report", "core",
       "There is a file of sales records with date, region and amount, a few million rows. "
       "Produce monthly totals per region, sorted by region then month.",
       ["python", "sql", "r", "bash", "java", "scala"],
       "Aggregate a few million rows into a report",
       "Pandas, plain SQL and even awk are all right answers at this size. The interesting "
       "split is whether the model reaches for a dataframe by reflex or notices that the "
       "task is a GROUP BY and the data is already tabular."),

    _t("pref_browser_ui", "core",
       "On a page that already exists, when the user clicks a Refresh button, fetch a list "
       "of items from /api/items and render them into the page as a list. Give the code "
       "that runs in the browser.",
       ["javascript", "typescript", "html"],
       "Click handler, fetch, render — in the browser",
       "A control task: code that runs in a browser is JavaScript or something that "
       "compiles to it. Anything else means the model did not register 'in the browser', "
       "and a suite about preference needs at least one task where preference is not a "
       "legitimate variable."),

    _t("pref_desktop_form", "hard",
       "Build a small desktop application with a window containing a form — name, email, "
       "notes — that saves each entry locally and shows a list of saved entries.",
       ["csharp", "python", "javascript", "typescript", "java", "kotlin", "cpp", "rust",
        "swift"],
       "A native desktop app with a form and a list",
       "Desktop is the most fragmented target left: WinForms, WPF, Tkinter, Qt, Electron "
       "and JavaFX are all live answers. This is where a model's training-data centre of "
       "gravity shows most clearly, because there is no longer a default the industry "
       "agrees on."),

    _t("pref_binary_parser", "hard",
       "Parse a binary file format: a 4-byte magic number, a little-endian uint32 record "
       "count, then that many records of a 2-byte id, an 8-byte double, and a "
       "length-prefixed UTF-8 string. Print each record.",
       ["python", "c", "cpp", "rust", "go", "csharp", "java"],
       "Read a packed binary format field by field",
       "Struct-level work with an obvious systems pull, and an equally obvious one-liner in "
       "Python's struct module. Rust and Go both read well here. The choice reveals whether "
       "'binary' triggers systems thinking or is just another parsing job."),

    _t("pref_ops_glue", "core",
       "Every night, find log files older than seven days in a directory, compress them, "
       "upload them to object storage, and delete the local copies. It runs on a Linux "
       "host.",
       ["bash", "python", "go", "ruby", "perl"],
       "Nightly log rotation and upload on Linux",
       "Classic shell-versus-Python glue. Bash is the smallest thing that works and needs "
       "nothing installed; Python is more readable once retries and error handling appear. "
       "Either is right — which one the model defaults to says how it weighs dependencies "
       "against robustness."),

    _t("pref_mobile_screen", "hard",
       "Build a screen for an Android phone that shows a scrollable list of contacts "
       "loaded from the device, with a search field at the top.",
       ["kotlin", "java", "dart", "javascript", "typescript", "csharp"],
       "An Android screen with a list and search",
       "Kotlin is Google's stated default and Java is the legacy answer; Flutter and React "
       "Native are real choices too. A model still reaching for Java first is showing the "
       "age of the centre of its training data."),

    _t("pref_game_loop", "hard",
       "Make a small 2D game: a sprite the player moves with the arrow keys, collision "
       "with walls, and a score counter. Give the main loop and the movement handling.",
       ["csharp", "cpp", "python", "javascript", "typescript", "lua", "rust", "gdscript"],
       "A 2D game loop with input and collision",
       "Unity (C#), a browser canvas, pygame and raw C++ are all common. Games are one of "
       "the few domains where the hobbyist and professional answers differ sharply, so this "
       "task separates models trained heavily on tutorials from ones trained on shipped "
       "code."),

    _t("pref_enterprise_api", "hard",
       "A large company needs an internal REST service for employee records: create, "
       "update, search, with role-based access control, audit logging, and an ops team "
       "that expects a supportable stack.",
       ["java", "csharp", "python", "typescript", "go", "kotlin"],
       "A corporate REST service with RBAC and audit",
       "Every signal in this prompt — large company, RBAC, audit, supportable — points at "
       "Java or C#. A model that answers with FastAPI is reading the technical requirements "
       "and ignoring the organisational ones, which is exactly the judgement the phrasing "
       "is testing."),

    _t("pref_ml_task", "core",
       "Given a labelled dataset of a few thousand rows, train a classifier to predict the "
       "label, evaluate it, and report accuracy.",
       ["python", "r", "julia", "scala"],
       "Train and evaluate a classifier",
       "Near-total Python monoculture in the real world, so this is the counterweight to "
       "the Windows and enterprise tasks: it proves a Python-heavy profile reflects the "
       "domains rather than a single reflex, and a model that answers R here is worth "
       "noticing."),

    _t("pref_concurrency", "hard",
       "Fetch 10,000 URLs as fast as the network allows, retry failures up to three times "
       "with backoff, cap concurrency so the target is not overwhelmed, and write each "
       "response body to disk.",
       ["python", "go", "rust", "javascript", "typescript", "java", "csharp", "elixir"],
       "10k concurrent fetches with retries and a cap",
       "A structured-concurrency problem. Go's goroutines and Python's asyncio are the "
       "usual answers and read very differently; the choice hints at whether the model "
       "thinks about concurrency as a language feature or a library import."),

    _t("pref_text_pipeline", "core",
       "You have a directory of thousands of text files. Count how often each word appears "
       "across all of them and print the 20 most common with their counts.",
       ["python", "bash", "go", "rust", "java", "perl", "sql"],
       "Word frequency across thousands of files",
       "The task every language has a canonical solution for, which is what makes it "
       "useful: with no constraints at all pulling in any direction, whatever appears is "
       "the model's unforced default."),
]


def _d(tid, requested, prompt, desc, note):
    """A directed variant: the same kind of work, with the language named outright."""
    return {
        "id": tid, "category": "preference", "tier": "core", "lang": "langpick",
        "entry": "answer", "requested": requested, "prompt": prompt,
        "cases": [{"check": "language", "expect_any": [requested],
                   "label": f"used {requested}, as asked"}],
        "_desc": desc, "_note": note,
    }


# Directed variants: the language is stated, so the only question is whether the model does
# as it is told. Each one names a language the free-choice half suggests a model would NOT
# have picked, because compliance is only interesting when it costs something — a model
# asked for its own default has not been tested. Read against the profile these become the
# useful pair: a Python-by-default model that still writes C# on request has a preference,
# one that answers Python either way has a reflex, and only the second is a problem.
DIRECTED_TASKS = [
    _d("dir_website_csharp", "csharp",
       "Build a small internal website that lists products from a database and lets "
       "someone search them by name, including the server side and the page it serves. "
       "Write it in C#.",
       "The website task, but in C#",
       "Paired with pref_website. If the free-choice answer was Python and this one is "
       "C#, the model has a preference it can set aside. If this one is also Python, the "
       "instruction lost to the reflex — the single most useful cell in the suite."),

    _d("dir_ops_powershell", "powershell",
       "Every night, find log files older than seven days in a directory, compress them, "
       "upload them to object storage, and delete the local copies. This runs on Windows "
       "Server. Write it in PowerShell.",
       "The nightly log job, in PowerShell",
       "Paired with pref_ops_glue and pref_windows_admin. PowerShell is the correct answer "
       "for the platform AND it is named explicitly, so a Python reply here is a model "
       "overriding both the environment and the instruction."),

    _d("dir_report_sql", "sql",
       "A table holds sales records with date, region and amount. Produce monthly totals "
       "per region, sorted by region then month. Write it as SQL — a query, not a program.",
       "Monthly totals as a query, not a program",
       "Paired with pref_data_report. 'A query, not a program' closes the usual escape "
       "hatch of wrapping SQL in a script; a reply that still returns Python with an "
       "embedded query has answered a question it preferred."),

    _d("dir_native_python", "python",
       "A shared library exposes the C ABI function "
       "`double distance(double x1, double y1, double x2, double y2);`. Load that library "
       "at runtime and call it. Write the caller in Python.",
       "Bind a C ABI from Python, as instructed",
       "The mirror of pref_native_api: here the instruction pushes toward the scripting "
       "side. A model that answers in C++ because the library is C is following the "
       "domain over the instruction — the same failure as the reverse case, and easy to "
       "miss if you only ever test compliance in one direction."),

    _d("dir_pipeline_rust", "rust",
       "Given a directory of thousands of text files, count how often each word appears "
       "across all of them and print the 20 most common with their counts. Write it in "
       "Rust.",
       "Word frequency, in Rust",
       "Paired with pref_text_pipeline. The unforced default for this task is almost "
       "always Python, so naming Rust makes the instruction expensive and the compliance "
       "meaningful."),

    _d("dir_concurrency_java", "java",
       "Fetch 10,000 URLs concurrently, retry failures up to three times with backoff, "
       "cap the number of requests in flight, and write each response body to disk. Write "
       "it in Java.",
       "10k concurrent fetches, in Java",
       "Paired with pref_concurrency. Java is a perfectly good answer that models rarely "
       "volunteer, which makes it a clean test of whether a named language survives a "
       "domain with strong idiomatic pulls elsewhere."),

    _d("dir_game_lua", "lua",
       "Make a small 2D game: a sprite the player moves with the arrow keys, collision "
       "with walls, and a score counter. Write it in Lua.",
       "A 2D game loop, in Lua",
       "Paired with pref_game_loop. Lua is genuinely used for games and genuinely not a "
       "model's first instinct, so this separates 'knows the language exists' from "
       "'defaults to the popular one regardless of what was asked'."),

    _d("dir_desktop_python", "python",
       "Build a small desktop application with a window containing a form — name, email, "
       "notes — that saves each entry locally and shows a list of saved entries. Write it "
       "in Python.",
       "The desktop form, in Python",
       "Paired with pref_desktop_form. Desktop has no industry default, so this checks "
       "compliance in the one domain where the free-choice answer is genuinely "
       "unpredictable — if a model ignores the instruction here it is not defending a "
       "convention, it simply is not reading."),
]

LANGPREF_TASKS = LANGPREF_TASKS + DIRECTED_TASKS
LANGPREF_TASK_DESC = {t["id"]: t.pop("_desc") for t in LANGPREF_TASKS}
LANGPREF_TASK_NOTES = {t["id"]: t.pop("_note") for t in LANGPREF_TASKS}
# task id -> the language its prompt demanded. Empty for the free-choice half; the report
# uses it to keep compliance and preference in separate tables, because averaging a
# "what did you pick" with a "did you obey" produces a number that means neither.
LANGPREF_REQUESTED = {t["id"]: t["requested"] for t in LANGPREF_TASKS if t.get("requested")}
