"""Token accounting.

Ollama exposes no ``/api/tokenize`` endpoint, and pulling a HuggingFace
tokenizer in just to count would add a heavyweight dependency for a number that
only needs to be right to within about 10%. So we estimate from character
length, and we *measure* the estimator instead of trusting it:
``python scripts/benchmark.py --calibrate`` compares this estimate against the
``prompt_eval_count`` Ollama reports for real prompts and prints the error, so
the constant below is a measured value rather than folklore.

It is deliberately biased slightly low-per-character (i.e. it over-estimates
token counts) so that budget arithmetic errs towards a shorter prompt, never
towards blowing the context window.
"""

from __future__ import annotations

#: Characters per token for English chat text under the Qwen2.5 BPE vocabulary.
#:
#: Calibrated, not guessed. ``benchmark.py --calibrate`` compared the estimate
#: against Ollama's reported ``prompt_eval_count`` over the real prompt mix: an
#: initial value of 3.6 over-estimated by +14.9%, implying a best fit of 4.14.
#: 4.0 is used instead of 4.14 so the estimator still errs ~3% high -- budget
#: arithmetic should shorten the prompt, never overflow the context window.
CHARS_PER_TOKEN = 4.0

#: Per-message chat-template overhead (role header + turn delimiters).
PER_MESSAGE_TOKEN_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN + 0.5))
