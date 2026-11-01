"""
DiAD entrypoint for MoViAD.
Follows the same pattern as other MoViAD entrypoints (patchcore.py, rd4ad.py).
"""

from dataclasses import dataclass, fields

from moviad.entrypoints.diad_unified import DiADUnifiedArgs, train_diad_unified


@dataclass(kw_only=True)
class DiADArgs(DiADUnifiedArgs):
    category: str


def train_diad(args: DiADArgs, logger=None) -> dict:
    values = {f.name: getattr(args, f.name) for f in fields(DiADUnifiedArgs)}
    values["categories"] = [args.category]
    per_category, _ = train_diad_unified(DiADUnifiedArgs(**values), logger)
    return per_category[args.category]
