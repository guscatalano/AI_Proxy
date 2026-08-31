"""langpref-v1: measuring what a model reaches for, not whether it works.

The grader's job is to name the language in a reply, so these tests are mostly about the
detector: fence tags, aliases, unlabelled blocks, and the case that matters most for this
suite — a web answer containing HTML, CSS and a server, where the language of the ANSWER is
the server, not the markup around it.
"""
import asyncio

from ai_proxy import bench_graders as G
from ai_proxy import bench_report as BR
from ai_proxy import proxy as P


def _task(tid):
    return next(t for t in P._BENCH_SUITES["langpref-v1"] if t["id"] == tid)


def _grade(tid, answer):
    return asyncio.run(P._bench_grade(answer, _task(tid), 30.0))


# ---- the detector --------------------------------------------------------------------------

def test_fence_tags_and_their_aliases():
    for tag, want in (("python", "python"), ("py", "python"), ("js", "javascript"),
                      ("node", "javascript"), ("c++", "cpp"), ("cpp", "cpp"),
                      ("cs", "csharp"), ("c#", "csharp"), ("golang", "go"),
                      ("rs", "rust"), ("ps1", "powershell"), ("sh", "bash"),
                      ("arduino", "cpp"), ("kt", "kotlin")):
        lang, how = G._bench_detect_language(f"here you go\n```{tag}\ncode here\n```")
        assert (lang, how) == (want, "fence"), (tag, lang)


def test_an_untagged_block_is_read_by_its_syntax():
    for code, want in (("#include <stdio.h>\nint main(){}", "cpp"),
                       ("using System;\nnamespace App { }", "csharp"),
                       ("package main\nfunc main() {}", "go"),
                       ("fn main() { let mut x = 1; }", "rust"),
                       ("public class Foo { System.out.println(1); }", "java"),
                       ("def run(x):\n    return x", "python"),
                       ("const a = 1;\nconsole.log(a)", "javascript"),
                       ("Get-Process | Write-Host", "powershell"),
                       ("<?php echo 1; ?>", "php")):
        lang, how = G._bench_detect_language(f"```\n{code}\n```")
        assert lang == want, (code[:20], lang)
        assert how == "syntax"


def test_the_server_is_the_answer_not_the_markup_around_it():
    """A web answer is usually HTML plus CSS plus the thing that serves them. The question
    this suite asks is which language the model chose to SOLVE the problem in."""
    reply = ("Here's the app.\n```html\n<html><body></body></html>\n```\n"
             "```css\nbody { margin: 0 }\n```\n"
             "```python\nfrom flask import Flask\napp = Flask(__name__)\n```")
    assert G._bench_detect_language(reply)[0] == "python"
    # ...but a pure front-end answer is still HTML, not "nothing".
    assert G._bench_detect_language("```html\n<html></html>\n```")[0] == "html"


def test_the_largest_block_wins_between_two_real_languages():
    reply = ("```python\nprint(1)\n```\n"
             "```go\n" + ("// a longer implementation\n" * 20) + "func main() {}\n```")
    assert G._bench_detect_language(reply)[0] == "go"


def test_prose_with_no_code_names_no_language():
    lang, how = G._bench_detect_language("I would use Python for this, it's well suited.")
    assert lang is None and how == "none"


# ---- grading: the pick is the payload, the score is the guardrail ---------------------------

def test_the_chosen_language_is_recorded_whatever_it_is():
    for answer, want in (("```python\nimport ctypes\n```", "python"),
                         ("```cpp\n#include <dlfcn.h>\n```", "cpp"),
                         ("```rust\nfn main() {}\n```", "rust")):
        g = _grade("pref_native_api", answer)
        assert g["picked"] == want
        assert g["passed"] == 1, "all three are defensible ways to call a C ABI"


def test_an_absurd_choice_for_the_domain_fails():
    """The score exists only for this: a browser task answered in C++ is not a preference,
    it is a misread."""
    g = _grade("pref_browser_ui", "```cpp\n#include <iostream>\nint main(){}\n```")
    assert g["passed"] == 0
    assert "odd fit" in g["cases"][0]["got"] and "cpp" in g["cases"][0]["got"]
    assert g["picked"] == "cpp", "the pick is still recorded — it is the data"
    # The languages that actually run in a browser pass.
    assert _grade("pref_browser_ui", "```javascript\nfetch('/api/items')\n```")["passed"] == 1
    assert _grade("pref_browser_ui", "```ts\nfetch('/api/items')\n```")["passed"] == 1


def test_an_answer_with_no_code_at_all_fails_rather_than_counting_as_a_pick():
    g = _grade("pref_website", "I'd recommend using Django for this.")
    assert g["passed"] == 0 and g["picked"] is None
    assert "no code" in g["cases"][0]["got"]


# ---- the suite itself ------------------------------------------------------------------------

def _free_choice():
    return [t for t in P._BENCH_SUITES["langpref-v1"] if not t.get("requested")]


def _directed():
    return [t for t in P._BENCH_SUITES["langpref-v1"] if t.get("requested")]


def test_no_free_choice_task_names_a_language(client):
    """A free-choice task that mentions a language measures the prompt, not the model. The
    directed half names one on purpose and is exempt."""
    banned = ("python", "java", "javascript", "typescript", "c#", "csharp", "golang",
              " rust", "kotlin", " php", "powershell", " bash", "flask", "django",
              "express", "asp.net", "pandas", "numpy", "unity", "react", "pygame",
              "spring", "laravel", ".py", ".cs", ".go")
    for t in _free_choice():
        low = t["prompt"].lower()
        for word in banned:
            assert word not in low, f"{t['id']} mentions {word!r} in its prompt"
    # ...and "C ABI"/"C++ library" phrasing is allowed only where the interface IS the task.
    assert "c abi" in _task("pref_native_api")["prompt"].lower()


def test_every_task_offers_a_real_choice(client):
    for t in _free_choice():
        opts = t["cases"][0]["expect_any"]
        assert len(opts) >= 3 or t["id"] == "pref_browser_ui", \
            f"{t['id']} has too few defensible options to be a preference question"
    for t in P._BENCH_SUITES["langpref-v1"]:
        assert t["lang"] == "langpick", t["id"]
        assert P._BENCH_TASK_DESC.get(t["id"]) and P._BENCH_TASK_NOTES.get(t["id"]), t["id"]
        assert P._BENCH_TASK_CATEGORY[t["id"]] == "preference", t["id"]
    names = {s["name"] for s in client.get("/__proxy/api/bench/suites").json()["suites"]}
    assert "langpref-v1" in names


# ---- the report is the deliverable ------------------------------------------------------------

def _run_with(model, picks):
    return {"id": model, "label": model, "model": model,
            "config": {"runs": 1, "suite": "langpref-v1"},
            "results": {"summary": {"quality": {}},
                        "rows": [{"task": t, "grade": {"picked": l}}
                                 for t, l in picks.items()]}}


def test_the_profile_shows_the_distribution_and_the_disagreements():
    a = _run_with("gemma4", {"pref_website": "python", "pref_windows_admin": "python",
                             "pref_throughput_service": "python",
                             "pref_embedded_blink": "cpp"})
    b = _run_with("qwen", {"pref_website": "typescript", "pref_windows_admin": "powershell",
                           "pref_throughput_service": "go", "pref_embedded_blink": "cpp"})
    html = BR._bench_language_profile_html([a, b], ["gemma4", "qwen"])
    assert "Language preference" in html
    assert "python" in html and "powershell" in html and "typescript" in html
    # the reflex call-out fires for the model that answered most tasks one way
    assert "answered 3 of 4 tasks in python" in html
    assert "answered" not in html.split("gemma4</b>")[-1].split("·")[0].replace(
        "answered 3 of 4 tasks in python", ""), "qwen has no dominant language to call out"
    # the task where both agreed is not highlighted; the ones they split on are
    grid = html.split("Task by task")[-1]
    assert 'class=win' in grid


def test_the_profile_is_absent_without_language_data():
    plain = {"id": "m", "label": "m", "model": "m", "config": {"runs": 1},
             "results": {"summary": {}, "rows": [{"task": "roman", "grade": {"passed": 1}}]}}
    assert BR._bench_language_profile_html([plain], ["m"]) == ""


# ---- directed variants: does a named language survive contact with a default? --------------

def test_a_directed_task_demands_exactly_the_language_it_names():
    for t in _directed():
        want = t["requested"]
        assert t["cases"][0]["expect_any"] == [want], t["id"]
        spoken = t["prompt"].lower().replace("c#", "csharp")
        assert want in spoken, f"{t['id']} asks for {want} without saying so in the prompt"
        assert P._BENCH_TASK_DESC.get(t["id"]) and P._BENCH_TASK_NOTES.get(t["id"]), t["id"]


def test_directed_compliance_is_graded_strictly():
    # Asked for C#, answered C#.
    g = _grade("dir_website_csharp", "```csharp\nusing System;\nvar app = 1;\n```")
    assert g["passed"] == 1 and g["picked"] == "csharp"
    # Asked for C#, answered Python — the reflex overriding the instruction.
    g2 = _grade("dir_website_csharp", "```python\nfrom flask import Flask\n```")
    assert g2["passed"] == 0 and g2["picked"] == "python"
    assert "odd fit" in g2["cases"][0]["got"]
    # Asked for SQL "a query, not a program"; a script wrapping a query is not compliance.
    g3 = _grade("dir_report_sql",
                "```python\nimport sqlite3\ndb.execute('SELECT r FROM sales')\n```")
    assert g3["passed"] == 0 and g3["picked"] == "python"
    assert _grade("dir_report_sql",
                  "```sql\nSELECT region, SUM(amount) FROM sales GROUP BY region\n```"
                  )["passed"] == 1


def test_the_directed_languages_cut_against_the_usual_default():
    """Compliance is only a test when it costs something: asking a model for the language
    it would have picked anyway measures nothing."""
    demanded = {t["requested"] for t in _directed()}
    assert "python" in demanded, "the reverse case matters too — see dir_native_python"
    assert len(demanded) >= 5, "too few distinct languages to separate obedience from luck"
    free_ids = {t["id"] for t in _free_choice()}
    for tid, twin in (("dir_website_csharp", "pref_website"),
                      ("dir_report_sql", "pref_data_report"),
                      ("dir_pipeline_rust", "pref_text_pipeline"),
                      ("dir_desktop_python", "pref_desktop_form")):
        assert twin in free_ids, f"{tid} has no free-choice twin to read it against"


def test_the_report_separates_obedience_from_preference():
    free = {"pref_website": "python", "pref_data_report": "python",
            "pref_text_pipeline": "python"}
    obedient = _run_with("gemma4", {**free, "dir_website_csharp": "csharp",
                                    "dir_report_sql": "sql"})
    stubborn = _run_with("qwen", {**free, "dir_website_csharp": "python",
                                  "dir_report_sql": "python"})
    html = BR._bench_language_profile_html([obedient, stubborn], ["gemma4", "qwen"])
    assert "Asked for" in html and "followed" in html
    assert "2/2" in html and "0/2" in html
    assert "qwen ignored it 2 times" in html
    # A directed task is not a free choice and must not pollute the preference profile.
    profile = html.split("Task by task")[0]
    assert "dir_website_csharp" not in profile
