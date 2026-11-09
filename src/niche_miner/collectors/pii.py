"""Scrub personal contact details and names of other people from collected text."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINK = re.compile(r"(?:https?://|www\.)\S+", re.I)
# Colombian mobiles (3xx) and landlines (60x), 10 digits, optional +57 — not salary amounts.
_PHONE = re.compile(r"(?<!\d)(?:\+?57[\s-]?)?\(?(?:3\d{2}|60\d)\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
# @handles (YouTube replies start with one, sometimes doubled: "@@usuario-x1").
_MENTION = re.compile(r"(?<![\w@])@+[\w.\-]*\w")


def scrub(text: str) -> str:
    """Emails, links, phone numbers and @mentions become placeholders."""
    text = _LINK.sub("[enlace]", _EMAIL.sub("[correo]", text))
    return _MENTION.sub("@usuario", _PHONE.sub("[teléfono]", text))
