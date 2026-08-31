"""Parsing of the customer-simulator's messages and classification of a
disclosed constraint into an attribute slot.

The local evaluator (evaluator/local_evaluator.py) emits a small, fixed set
of message templates. We parse them when they match (precise,
high-confidence signal) and always also fall back to raw keyword extraction
from the whole message, so the agent keeps working against a private
evaluator that may paraphrase the wording.
"""

from __future__ import annotations

import re

RE_CATEGORY = re.compile(r"^I'm looking for ([^.,]+)[.,]", re.IGNORECASE)
RE_KEY_REQUIREMENT = re.compile(
    r"A key requirement is:\s*(.+?)\.?\s*$", re.IGNORECASE | re.DOTALL
)
RE_REVEALED = re.compile(
    r"For that, what matters is:\s*(.+?)\.?\s*$", re.IGNORECASE | re.DOTALL
)
RE_OVERRIDE = re.compile(
    r"Actually, ignore my earlier preference\.\s*What I need is:\s*(.+?)\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
RE_NO_PREFERENCE = re.compile(
    r"I don't have (?:an additional preference|a preference)", re.IGNORECASE
)
RE_GENERIC_REBUFF = re.compile(r"not quite right yet", re.IGNORECASE)

# Turn-1 opener shape: "I'm looking for <category>.<tail>". The tail is empty
# for a bare opener, "A key requirement is: ..." for a Buying opener (handled
# by RE_KEY_REQUIREMENT), "but I'm still exploring." for a Browsing opener
# (no constraint -- ignored via RE_OPENER_FILLER), and a verbatim
# intent-card preference for an Intent-Override opener
# ("I'm looking for <cat>. <old_value>"). That last case is a real,
# catalog-derived attribute of the target (soft_preferences[-1]) and is worth
# capturing turn 1 instead of only reaching the rerank once the simulator
# re-discloses it several turns later. README: "Opener trailing clause".
RE_OPENER_TAIL = re.compile(r"^I'm looking for [^.,]+[.,]\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
RE_OPENER_FILLER = re.compile(r"^(?:but\s+)?I'm still exploring\b", re.IGNORECASE)

# Simulator-synthetic constraint phrases: the local evaluator builds these
# from intent-card metadata with a "<attr>: <value>" or "budget around $..."
# shape the catalog text does not use verbatim. Excluded from the EXACT
# whole-field match term of Agent._rerank.
_SYNTHETIC_PHRASE_RE = re.compile(
    r"^(?:color|size|material|budget|style|use[_ ]?case|feature|brand|department)\b\s*:?"
    r"|^budget\s+around\s+\$",
    re.IGNORECASE,
)

# Mirrors the local evaluator's own classify_constraint() bucketing. Used to
# detect same-attribute conflicts on override (e.g. an old "color: blue"
# next to a new override value "red").
_MATERIAL_WORDS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
_COLOR_WORDS = (
    "color",
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
_STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
_USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")


def _classify(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in _MATERIAL_WORDS):
        return "material"
    if any(word in lowered for word in _COLOR_WORDS):
        return "color"
    if any(word in lowered for word in _SIZE_WORDS):
        return "size"
    if any(word in lowered for word in _STYLE_WORDS):
        return "style"
    if any(word in lowered for word in _USE_CASE_WORDS):
        return "use_case"
    return "feature"


# Attribute-keyed slots for the dialogue state machine. "category" is
# tracked separately (state.category_text) since it comes from its own regex
# and its own ask_attribute value; everything _classify() can return gets a
# real slot here.
_SLOT_ATTRS = (
    "material",
    "color",
    "size",
    "style",
    "budget",
    "use_case",
    "feature",
    "brand",
)

# Slots where a same-bucket conflict is reliably detectable (narrow,
# unambiguous keyword sets) -- safe to erase-and-rewrite on override.
# "feature"/"style"/"use_case" have loose keyword matching and are excluded:
# for those, override behaves as pure accumulation.
_SAFE_CONFLICT_ATTRS = {"material", "color", "size", "budget"}
