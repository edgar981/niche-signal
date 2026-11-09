"""robots.txt rules with the * and $ wildcards (urllib.robotparser treats them literally)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit


def _pattern(rule: str) -> re.Pattern[str]:
    anchored = rule.endswith("$")
    body = re.escape(rule[:-1] if anchored else rule).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


class Robots:
    """Allow/Disallow rules from the groups for every crawler (User-agent: *).
    The longest matching rule wins and Allow wins a tie, as in RFC 9309."""

    def __init__(self, text: str):
        self.rules: list[tuple[int, bool, re.Pattern[str]]] = []
        agents: list[str] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:  # a User-agent line after rules starts a new group
                    agents, in_rules = [], False
                agents.append(value)
            elif key in ("allow", "disallow"):
                in_rules = True
                if "*" in agents and value:  # an empty Disallow allows everything
                    self.rules.append((len(value), key == "allow", _pattern(value)))

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best = (-1, True)
        for length, allow, pattern in self.rules:
            if pattern.match(path) and (length, allow) > best:
                best = (length, allow)
        return best[1]
