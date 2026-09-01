"""Reading the customer: turn the simulator's message templates back into
constraint phrases.

Known phrasings are matched exactly; anything unrecognised falls through to
keeping the whole line as a weak phrase, so rewording degrades the signal
instead of erasing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import _tidy

OPENING_BUYING = re.compile(
    r"^i'm looking for (?P<category>.+?)\.\s*a key requirement is:\s*(?P<constraint>.+?)\.?$",
    re.I,
)
OPENING_BROWSING = re.compile(
    r"^i'm looking for (?P<category>.+?),\s*but i'm still exploring\.?$", re.I
)
OPENING_OVERRIDE = re.compile(
    r"^i'm looking for (?P<category>.+?)\.\s*(?P<constraint>.+?)$", re.I
)
REVEAL = re.compile(r"^for that, what matters is:\s*(?P<body>.+?)\.?$", re.I)
OVERRIDE = re.compile(
    r"^actually, ignore my earlier preference\.\s*what i need is:\s*(?P<constraint>.+?)\.?$",
    re.I,
)
NO_INFO = re.compile(
    r"^(i don't have (a preference for|an additional preference for)|"
    r"those options are not quite right)",
    re.I,
)
# Two refusals that read alike but mean different things. "a preference" is a
# one-off Boundary-scenario shrug and says nothing about the attribute; "an
# additional preference" means that attribute is genuinely used up.
NO_PREFERENCE = re.compile(
    r"^i don't have (?P<kind>a|an additional) preference for (?P<attribute>[a-z_]+)",
    re.I,
)

ALLOWED_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)


def _split_reveal(body: str) -> list[str]:
    """The customer joins constraints with '; ', but a constraint may itself
    contain a semicolon. Splitting is safe under both readings: if the real
    constraint was 'A; B', then A and B are each still phrases of the target,
    so we never emit the ambiguous joined form -- it matches the wrong product
    often enough to evict the right one."""
    parts = [_tidy(part) for part in body.split(";")]
    parts = [part for part in parts if part]
    return parts or [_tidy(body)]


@dataclass
class Turn:
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    is_override: bool = False
    informative: bool = True
    refused: str | None = None
    refusal_is_boundary: bool = False


def parse_message(message: str) -> Turn:
    """Recover constraint phrases from one customer turn.

    Known phrasings are matched exactly; anything unrecognised falls through to
    keeping the whole line as a weak phrase, so rewording degrades the signal
    instead of erasing it.
    """
    text = re.sub(r"\s+", " ", str(message or "")).strip()
    if not text:
        return Turn(informative=False)

    match = OVERRIDE.match(text)
    if match:
        # The superseded preference still came from the target product, so we
        # never retract it -- we only add what the override discloses.
        return Turn(constraints=[_tidy(match.group("constraint"))], is_override=True)

    match = REVEAL.match(text)
    if match:
        return Turn(constraints=_split_reveal(match.group("body")))

    match = NO_PREFERENCE.match(text)
    if match:
        return Turn(
            informative=False,
            refused=match.group("attribute").lower(),
            refusal_is_boundary=match.group("kind").lower() == "a",
        )

    if NO_INFO.match(text):
        return Turn(informative=False)

    match = OPENING_BUYING.match(text)
    if match:
        return Turn(
            category=_tidy(match.group("category")),
            constraints=[_tidy(match.group("constraint"))],
        )

    match = OPENING_BROWSING.match(text)
    if match:
        return Turn(category=_tidy(match.group("category")))

    match = OPENING_OVERRIDE.match(text)
    if match:
        return Turn(
            category=_tidy(match.group("category")),
            constraints=[_tidy(match.group("constraint"))],
        )

    return Turn(constraints=[_tidy(text)])
