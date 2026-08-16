"""longctx-v1: does a long context window actually hold anything?

Every other suite here fits in a few thousand tokens, so none of them can tell the difference
between a model that advertises a million-token window and one that can use it. That gap is
not academic: nemotron-3.5-lightning pins at 1,048,576 tokens on this box and prefills 700,122
of them without complaint, and nothing in the benchmark could say whether the model could
still read what it had been given.

The first attempt to answer that by hand got it wrong, which is the reason this suite exists
rather than a script. Two ad-hoc runs at 700k scored 2/5 and 3/5 and led to "300k is the
usable window" — a claim built on three defects since fixed: a scorer that read only `content`
when the answer was in `reasoning`, a 2,048-token output budget against reasoning blocks
later measured at 6,134 characters, and a flat 600s client timeout that aborted the request
mid-prefill. Run properly, this ladder scored 149 of 150 needles from 16k through 512k.

So: plant five facts at fixed depths through a haystack of a known size, ask for all five in
one request, and score what comes back. One request per size, five cases per request, which
keeps a metric this expensive affordable — a 300k-token prefill costs about three minutes of
exclusive GPU on a GB10.

Three things this suite does that the obvious version gets wrong:

* The first needle sits near the START. Ollama truncates a too-long prompt from the front, so
  a model whose real window is smaller than the prompt silently loses the beginning and can
  still answer everything else. A needle at the end proves nothing; the early one is the
  load-bearing case.

* A WRONG answer is scored apart from a MISSING one, and both apart from silence. The one
  miss in the first clean run was a model writing `delta=MISSING` — it had lost the fact and
  said so, which is a different state from confidently reporting another needle's code.
  Averaging them into "not correct" hides the distinction that says which is happening.

* Haystacks are generated, never stored. A 300k-token prompt is 1.2 MB of text and the ladder
  runs to 1M; as module literals that is tens of megabytes loaded into every process that
  imports the proxy. They are built on first read and cached (see _LazyTask), so importing
  this module costs nothing for the runs that do not use it — which is all of them, most days.

The filler is deliberately repetitive — enumerated ledger lines that all look alike. That is
the hard case for binding a fact to a position, and it makes this a lower bound rather than a
flattering number. A real 700k context is more distinctive than this one.
"""

# (name, code, where it goes as a fraction through the haystack)
NEEDLES = [
    ("alpha",   "CRIMSON-4417",    0.00),
    ("bravo",   "MERIDIAN-8823",   0.25),
    ("charlie", "OBSIDIAN-1596",   0.50),
    ("delta",   "TANGERINE-7304",  0.75),
    ("echo",    "PERIWINKLE-2051", 1.00),
]

_WORDS = ("ledger reconciliation batch transfer settlement clearing custody escrow "
          "remittance drawdown accrual amortisation covenant tranche waterfall").split()

# Measured against this model family's tokeniser on a 2,000-line sample: 60,786 tokens, so
# 30.39 per line. Only used to pick a line count for a target size — the run records the
# upstream's own prompt_tokens, which is the number that gets reported.
_TOKENS_PER_LINE = 30.39

QUESTION = ("\n\nFive lines in the archive above are vault access codes. List all five, one "
            "per line, in the form NAME=CODE. If you cannot find one, write NAME=MISSING. "
            "Reply with only those five lines.")


def _filler(i: int) -> str:
    return (f"{i:07d} | archive record {i:07d}: "
            + " ".join(_WORDS[(i + k) % len(_WORDS)] for k in range(6)) + ".")


def haystack(n_lines: int, seed: int = 0) -> str:
    """One haystack. The same `seed` always gives the same text, so a re-run of a model is
    comparable with the first rather than merely similar to it.

    Different seeds give a genuinely different draw at the same depths, which is the whole
    point of repeating a rung. Repeating the IDENTICAL prompt five times would measure the
    prompt cache: the backend serves repeats two through five off a KV prefix it already has,
    in a fraction of the time, and five matching answers would look like consistency when they
    are one answer counted five times. Varying the very first line guarantees no shared prefix.

    What varies is the filler and where each needle lands. What does NOT vary is the codes or
    the depths — the depths are the axis being measured, and the codes have to stay fixed
    because the grading cases carry them.
    """
    off = seed * 7919                      # coprime with the word list, so every line shifts
    lines = [_filler(i + off) for i in range(n_lines)]
    for idx, (name, code, frac) in enumerate(NEEDLES):
        if idx == 0:
            # The first needle stays near the top on every seed: it is the case that proves the
            # window, because a backend that front-truncates loses this one first. Line 3 rather
            # than line 0 — at the very top it sits where a chat template's preamble goes, and
            # reads as an instruction rather than as content.
            pos = 3
        else:
            # Jitter around the target depth so five repeats do not all probe the identical
            # absolute offset, while staying close enough that the depth label stays honest.
            jitter = int(n_lines * 0.02 * (((seed * 31 + idx * 17) % 7) - 3) / 3)
            pos = min(max(4, int(n_lines * frac) + jitter), n_lines - 1)
        lines[pos] = f"The {name} vault access code is {code}."
    return "\n".join(lines) + QUESTION


class _LazyTask(dict):
    """A task whose prompt is built the first time it is read.

    The runner reads task["prompt"], and these prompts are up to 1.2 MB each. Building them
    at import would add megabytes and a measurable delay to every process that imports the
    proxy — including the one serving traffic, which will never run this suite.
    """

    def __init__(self, n_lines, **kw):
        super().__init__(**kw)
        self._n_lines = n_lines

    def __getitem__(self, key):
        if key == "prompt" and "prompt" not in self:
            super().__setitem__("prompt", haystack(self._n_lines))
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "prompt":
            return self["prompt"]
        return super().get(key, default)

    def prompt_for(self, seed: int):
        """A zero-argument builder for repeat `seed`, NOT the built prompt.

        The runner assembles every unit of a run before sending the first one. Materialising
        eight rungs × five repeats up front would hold something like a hundred megabytes of
        filler in memory for the whole run; returning a thunk keeps one prompt alive at a time.
        """
        n = self._n_lines
        return lambda: haystack(n, seed)


def _cases():
    return [{"check": "needle", "name": n, "code": c,
             "depth": "start" if f == 0.0 else "end" if f == 1.0 else f"{int(f * 100)}%",
             "label": f"{n} @ {'start' if f == 0.0 else 'end' if f == 1.0 else str(int(f * 100)) + '%'}"}
            for n, c, f in NEEDLES]


def _task(target_tokens: int, label: str):
    return _LazyTask(
        int(target_tokens / _TOKENS_PER_LINE),
        id=f"longctx_{label}",
        category="longcontext", tier="hard", lang="needles", entry="",
        target_tokens=target_tokens,
        cases=_cases(),
    )


# The ladder runs from "fits anywhere" to "the largest window anything here advertises", so the
# result is a curve rather than a point, and so the same suite says something useful about a 32k
# model and a 1M one. The runner drops the rungs that do not fit the model under test and records
# them in env.skipped_context — the same contract as a missing toolchain, because scoring a model
# zero on a prompt its window cannot hold measures the configuration, not the model.
#
# Rungs are chosen where the answer changes, not on a round grid: 16k is the control, 128k is
# where most local deployments sit, 262,144 is the Ollama server default this box ran for months,
# 300k is the first rung past it, and 512k/700k/1M cover the range where a "1M" claim is actually
# being tested. Measured here, 300k scored 5/5 and 700k scored 2/5 — a cliff that a ladder
# stopping at 300k would have reported as a clean sweep.
LONGCTX_TASKS = [
    _task(16_000, "16k"),
    _task(64_000, "64k"),
    _task(128_000, "128k"),
    _task(262_144, "256k"),
    _task(300_000, "300k"),
    _task(512_000, "512k"),
    _task(600_000, "600k"),
    _task(700_000, "700k"),
    _task(800_000, "800k"),
    _task(900_000, "900k"),
    _task(1_000_000, "1m"),
]

# The deep end on its own. Re-running six rungs that already scored 149/150 to reach the two
# that are actually in question costs an hour of exclusive GPU for no information, and this
# ladder is expensive enough that the difference matters.
LONGCTX_DEEP_TASKS = [t for t in LONGCTX_TASKS if t["target_tokens"] >= 600_000]

# The other half. Together the two cover the ladder without overlap, so a comparison that needs
# only one end of it does not pay for the other — the deep rungs are four hours of exclusive
# GPU and the shallow ones are barely one.
LONGCTX_LITE_TASKS = [t for t in LONGCTX_TASKS if t["target_tokens"] < 600_000]

LONGCTX_TASK_DESC = {t["id"]: f"Recall five facts planted through a "
                              f"{t['target_tokens']:,}-token haystack."
                     for t in LONGCTX_TASKS}

LONGCTX_TASK_NOTES = {
    "longctx_16k": "The control. A model that misses needles here has a retrieval problem, "
                   "not a long-context one.",
    "longctx_64k": "Comfortably inside every window on this box.",
    "longctx_128k": "Where most local deployments are configured.",
    "longctx_256k": "The Ollama server default (262,144). The last rung most setups can reach "
                    "without deliberately raising the window.",
    "longctx_300k": "The first rung past the default. Measured 5/5 on nemotron-3.5-lightning.",
    "longctx_512k": "Half the advertised million. Between the last known-good rung and the "
                    "first known-bad one, so this is where the cliff gets located.",
    "longctx_600k": "The first rung above the last clean sweep (512k scored 25/25).",
    "longctx_700k": "Two early one-off runs scored 2/5 and 3/5 here, but all three were run "
                    "through defects since fixed — a scorer that read only `content` when the "
                    "answer was in `reasoning`, a 2,048-token budget against reasoning blocks "
                    "measured at 6,134 characters, and a flat 600s client timeout that aborted "
                    "the request mid-prefill. Treat those numbers as unmeasured.",
    "longctx_800k": "Past anything measured on this box.",
    "longctx_900k": "The highest rung that fits: a 1M window minus room to answer, minus the "
                    "estimator's 10% headroom, leaves about 936k.",
    "longctx_1m": "The advertised maximum, and unreachable by construction: a prompt cannot "
                  "fill the whole window and still leave room for a reply. Kept as the rung "
                  "that documents the ceiling — it is skipped and recorded, never failed.",
}
