"""An LLM pass over the reranked top-10.

Why this layer exists
---------------------
The linear reranker in `agent.py` decides a slate with a dot product over six
scalar features. Where the customer's disclosed evidence separates candidates,
that is enough and this layer has nothing to add. Where it does not -- and it
usually does not on turn 1, because the customer has disclosed exactly one
constraint and hundreds of catalog rows satisfy it identically -- the ordering
falls through to popularity, ratings velocity and BM25 position, three priors
that know nothing about what the shopper actually said.

An LLM reads the candidates as products rather than as feature vectors, so it
can act on the parts of the text the scalar features flatten away: that a
constraint lands on the item's defining material rather than on its carry bag,
that "Men's" and "Ladies" are different products, that a title's head noun is
or is not the thing being asked for.

What it is allowed to do
------------------------
Reorder. Nothing else. The candidate list it returns is a permutation of the
list it was given, enforced here rather than trusted from the model: indices it
omits are appended in their incoming order, indices it invents or repeats are
dropped. The model cannot introduce a product, cannot drop one, and therefore
cannot turn a hit into a miss -- the worst it can do is order a slate badly,
which is the same thing the linear model risks.

Failure is always backwards-compatible
--------------------------------------
No client, an unreachable server, a timeout, a malformed reply, an exception
anywhere: every one of these returns the incoming order unchanged. Official
scoring may run with network access disabled (`docs/submission_rules.md`), so
the offline path is not a degraded mode, it is the expected one -- the agent
scores exactly what it scored before this file existed.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# "ollama", "anthropic", or "off". Read once at construction.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_VERSION = "2023-06-01"

# A turn has to finish; the evaluator discards the whole response if respond()
# raises, and a slow model must not cost us the recommendations we already
# computed. Kept short deliberately -- a timeout is a fallback, not an error.
TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT", "20"))
MAX_OUTPUT_TOKENS = 64

# How much of each candidate the model sees. Long enough to carry the bullet a
# constraint was quoted from, short enough that ten of them stay readable.
TITLE_CHARS = 120
FEATURE_CHARS = 100
MAX_FEATURES = 5


# What the model is asked to produce.
#
# "pick" asks for one number, "rank" for a full ordering. They are not equally
# hard: a full ranking is nine comparative judgements and a small model spends
# most of them badly, while the exposure turns only ever consume the first
# element. Under "pick" the model chooses a single candidate to promote and
# everything else keeps the linear order, so a bad answer costs one position
# rather than scrambling the slate.
PROMPT_MODE = os.environ.get("LLM_PROMPT_MODE", "pick").strip().lower()

# How far down the linear slate the model's promotion is allowed to reach.
#
# A ceiling on the damage. The linear reranker is right about the top-1 roughly
# two thirds of the time on contested slates, so a model that reaches to rank 9
# is usually overruling a better guess than its own. 0 disables the ceiling.
MAX_PROMOTION_RANK = int(os.environ.get("LLM_MAX_PROMOTION", "0") or 0)


RANK_INSTRUCTION = (
    "Answer with candidate numbers only, best first, comma separated, no other "
    "text. Example: 4,1,7"
)
PICK_INSTRUCTION = (
    "Answer with the number of the single best candidate and nothing else. "
    "Example: 4"
)


SYSTEM_PROMPT = (
    "You rank shopping search results. The shopper is describing one specific "
    "product they already have in mind, quoting phrases from that product's own "
    "listing. Your job is to decide which candidate they are describing.\n\n"
    "Judge on these, in order:\n"
    "1. Does the quoted requirement describe what the product IS, or something "
    "incidental to it? A jacket whose shell is nylon matches \"nylon\"; a cotton "
    "jacket that ships with a nylon carry bag does not.\n"
    "2. Does the product type match the category the shopper named?\n"
    "3. Do gender, size, and intended use fit everything the shopper has said?\n"
    "Popularity is not a reason to rank a product higher.\n\n"
) + (PICK_INSTRUCTION if PROMPT_MODE == "pick" else RANK_INSTRUCTION)


def build_prompt(category: str | None, evidence: list[str], cards: list[dict]) -> str:
    """The user turn: what the shopper disclosed, then the candidates."""
    lines = []
    lines.append(f"Shopper is looking for: {category or 'unspecified'}")
    if evidence:
        lines.append("Requirements they have stated, quoted verbatim:")
        lines.extend(f"  - {item}" for item in evidence)
    else:
        lines.append("They have not stated any specific requirement yet.")
    lines.append("")
    lines.append("Candidates:")
    for number, card in enumerate(cards, 1):
        lines.append(f"{number}. {card.get('title', '')}")
        store = card.get("store")
        if store:
            lines.append(f"   brand: {store}")
        for feature in card.get("features", []):
            lines.append(f"   - {feature}")
    lines.append("")
    if PROMPT_MODE == "pick":
        lines.append(
            f"Which one of these {len(cards)} candidates is the shopper "
            "describing? Answer with its number only."
        )
    else:
        lines.append(
            f"Rank all {len(cards)} candidates, best match first. Numbers only."
        )
    return "\n".join(lines)


def parse_order(reply: str, count: int, mode: str | None = None) -> list[int]:
    """The model's reply as a permutation of range(count).

    Lenient on purpose: the model is asked for "4,1,7" but may return prose, a
    numbered list, or duplicates. Every integer in range is taken in the order
    it first appears, and whatever the model left out keeps its incoming order
    at the back. The result is always a full permutation, so a garbled reply
    degrades toward the linear ordering instead of corrupting the slate.

    In "pick" mode only the first number is read, and it is promoted to the
    front of an otherwise untouched linear order. The rest of the reply is
    discarded rather than parsed -- a small model asked for one number often
    keeps talking, and the numbers in its explanation are not a ranking.
    """
    mode = mode or PROMPT_MODE
    found: list[int] = []
    for token in re.findall(r"\d+", reply or ""):
        index = int(token) - 1
        if 0 <= index < count and index not in found:
            found.append(index)
            if mode == "pick":
                break

    if mode == "pick" and found:
        # A promotion from deep in the slate is more likely a misread than an
        # insight, so the ceiling declines it and leaves the linear order.
        if MAX_PROMOTION_RANK and found[0] >= MAX_PROMOTION_RANK:
            return list(range(count))
        chosen = found[0]
        return [chosen] + [index for index in range(count) if index != chosen]

    found.extend(index for index in range(count) if index not in found)
    return found


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class OllamaClient:
    """A local model over Ollama's chat endpoint. No key, no network egress."""

    name = "ollama"

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as response:
                names = {
                    entry.get("name") for entry in json.loads(response.read()).get("models", [])
                }
        except Exception:
            return False
        # Ollama accepts "llama3.2" for "llama3.2:latest", so compare on the
        # bare name too rather than rejecting a model that is actually present.
        bare = {str(item).split(":", 1)[0] for item in names}
        return self.model in names or self.model.split(":", 1)[0] in bare

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Temperature 0 keeps a run reproducible, which the rest of the
            # harness already is; anything else makes a measured delta noise.
            "options": {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
        }
        data = _post(f"{self.host}/api/chat", payload, {}, TIMEOUT_SECONDS)
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        }
        return (data.get("message", {}).get("content") or ""), usage


class AnthropicClient:
    """The Messages API. Requires ANTHROPIC_API_KEY in the environment."""

    name = "anthropic"

    def __init__(self, base: str = ANTHROPIC_BASE, model: str = ANTHROPIC_MODEL) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _post(
            f"{self.base}/v1/messages", payload,
            {"x-api-key": self.key, "anthropic-version": ANTHROPIC_VERSION},
            TIMEOUT_SECONDS,
        )
        text = "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )
        raw = data.get("usage", {})
        usage = {
            "prompt_tokens": int(raw.get("input_tokens") or 0),
            "completion_tokens": int(raw.get("output_tokens") or 0),
        }
        return text, usage


def make_client(provider: str | None = None):
    """The configured client, or None when the model is off or unreachable.

    None is not an error state. It is the offline path, and the agent runs its
    deterministic pipeline unchanged on it.
    """
    choice = (provider or LLM_PROVIDER).strip().lower()
    if choice in ("", "off", "none", "false", "0"):
        return None
    client = {"ollama": OllamaClient, "anthropic": AnthropicClient}.get(choice)
    if client is None:
        return None
    try:
        instance = client()
        return instance if instance.available() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The reranker
# ---------------------------------------------------------------------------

class LLMReranker:
    """Reorders a slate. Never changes its membership."""

    def __init__(self, client=None, provider: str | None = None) -> None:
        self.client = client if client is not None else make_client(provider)
        self.calls = 0
        self.failures = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def order(
        self, category: str | None, evidence: list[str], cards: list[dict]
    ) -> tuple[list[int], dict]:
        """Indices of `cards`, best first, plus this call's token usage.

        Returns the identity order on every failure path, so the caller can
        apply the result unconditionally.
        """
        blank = {"prompt_tokens": 0, "completion_tokens": 0}
        identity = list(range(len(cards)))
        # One candidate has nothing to reorder, and zero has nothing to send.
        if not self.enabled or len(cards) < 2:
            return identity, blank
        try:
            prompt = build_prompt(category, evidence, cards)
            reply, usage = self.client.complete(SYSTEM_PROMPT, prompt)
            self.calls += 1
            self.usage["prompt_tokens"] += usage["prompt_tokens"]
            self.usage["completion_tokens"] += usage["completion_tokens"]
            if not reply.strip():
                self.failures += 1
                return identity, usage
            return parse_order(reply, len(cards)), usage
        except Exception:
            self.failures += 1
            return identity, blank
