"""
geoinr_faults/features/strati.py
------------------------
Small configuration object for defining stratigraphic groups ("strati").

Each strati contains an ordered list of formations (top -> bottom). Different
strati are treated as independent stacks: ordering constraints are applied
within each strati but not across strati.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class StratiConfig:
    """
    One independent stratigraphic stack.

    Parameters
    ----------
    name       : Optional identifier for display/debugging.
    formations : Ordered formation names (top -> bottom) in this stack.
    """

    name: str = ""
    formations: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "StratiConfig":
        name = d.get("name", "")
        formations = d.get("formations", d.get("horizon_formations"))
        if formations is None:
            formations = []
        return cls(name=name, formations=list(formations))


