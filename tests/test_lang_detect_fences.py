"""Which fenced block counts as "the language the model chose".

Every case here is taken from a real langpref-v1 answer. The detector used to split fences
into just real-vs-scaffolding, which produced two failures: an ops answer made of a full
bash script plus a two-line crontab resolved to `cron` (bash was demoted as scaffolding, so
the crontab was the only "real" candidate left and won by default), and a REST design
answered with a Mermaid diagram was recorded as preferring `mermaid`. Both scored the model
wrong and both polluted the preference distribution with things that are not languages.
"""
from ai_proxy import bench_graders as G


def fence(tag, body="x = 1\n"):
    return "```%s\n%s```\n" % (tag, body)


def test_a_shell_script_beside_a_crontab_is_bash():
    """The actual gemma4 answer to pref_ops_glue: a long bash script, then the crontab line
    that schedules it. It was scored 0/1 for 'choosing cron'."""
    answer = ("The most robust approach is a Bash script scheduled with cron.\n"
              + fence("bash", "#!/bin/bash\n" + "echo rotate\n" * 40)
              + "Then install the schedule:\n"
              + fence("cron", "0 3 * * * /usr/local/bin/log_archive.sh\n"))
    assert G._bench_detect_language(answer)[0] == "bash"


def test_a_crontab_alone_is_not_a_language():
    assert G._bench_detect_language(fence("cron", "0 3 * * * /bin/true\n"))[0] is None


def test_a_mermaid_diagram_is_not_a_language():
    """qwen3.6 on pref_enterprise_api: a diagram and a JSON sample, no code. The honest
    answer is 'no identifiable code', not 'prefers mermaid'."""
    answer = fence("mermaid", "graph TD;A-->B;\n") + fence("json", '{"role": "admin"}\n')
    assert G._bench_detect_language(answer)[0] is None


def test_a_prose_only_answer_detects_nothing():
    assert G._bench_detect_language(fence("text", "We would use a layered design.\n"))[0] is None


def test_a_real_language_after_a_text_block_still_wins():
    """nemotron on the same task got this right already — it must stay right."""
    answer = fence("text", "Design notes\n") + fence("java", "class Api {}\n" * 20)
    assert G._bench_detect_language(answer)[0] == "java"


def test_html_scaffolding_loses_to_the_server_language():
    answer = fence("html", "<html></html>\n" * 30) + fence("python", "app = Flask(__name__)\n")
    assert G._bench_detect_language(answer)[0] == "python"


def test_html_alone_still_counts_when_it_is_the_answer():
    """Demoted is not excluded — a browser task answered in HTML has still made a choice."""
    assert G._bench_detect_language(fence("html", "<canvas></canvas>\n"))[0] == "html"


def test_powershell_is_never_demoted():
    """It is the right answer to the Windows-ops task and a directed target, so it must not
    be treated as packaging the way bash is: a short PowerShell answer beats a long shell
    block, which is exactly the comparison bash would have lost."""
    answer = fence("powershell", "Get-ChildItem\n") + fence("bash", "echo hi\n" * 40)
    assert G._bench_detect_language(answer)[0] == "powershell"


def test_size_decides_between_two_real_languages_not_tier():
    """Both real: the bigger block is the answer, whatever it happens to be."""
    answer = fence("powershell", "Get-ChildItem\n") + fence("python", "import os\n" * 30)
    assert G._bench_detect_language(answer)[0] == "python"


def test_the_larger_of_two_real_languages_wins():
    answer = fence("go", "package main\n") + fence("rust", "fn main() {}\n" * 40)
    assert G._bench_detect_language(answer)[0] == "rust"


def test_detection_reports_how_it_decided():
    assert G._bench_detect_language(fence("python"))[1] == "fence"


def test_data_formats_do_not_mask_a_real_answer():
    answer = (fence("yaml", "a: 1\n" * 50) + fence("json", "{}\n" * 50)
              + fence("csharp", "class C {}\n"))
    assert G._bench_detect_language(answer)[0] == "csharp"
