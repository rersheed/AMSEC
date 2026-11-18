"""Explicit dataset alignment policies (never silent app-id pairing)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentPolicy:
    name: str
    description: str


NIDS_ONLY = AlignmentPolicy(
    name="nids_only",
    description=(
        "Cell 2/4 scored runs use NF-UQ-NIDS rows only. API is optional and was missing "
        "in the executed cell-4 Kaggle session."
    ),
)

API_SELF_ALIGN = AlignmentPolicy(
    name="api_self_truncate",
    description=(
        "Cell 22 truncates API static features, sequences, and labels to min(len). "
        "NIDS may be loaded but is not fused. No shared application identifier exists."
    ),
)

FORBIDDEN_ROW_JOIN = AlignmentPolicy(
    name="forbidden_independent_row_join",
    description=(
        "Do not concatenate or zip independently sourced API and NIDS tables by row index "
        "and claim they represent the same Android application."
    ),
)
