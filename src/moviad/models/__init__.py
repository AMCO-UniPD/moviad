"""Public model exports, loaded only when they are first requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "CFA": (".cfa.cfa", "CFA"),
    "Ganomaly": (".ganomaly.ganomaly", "Ganomaly"),
    "Padim": (".padim.padim", "Padim"),
    "PadimLite": (".padim.padim_lite", "PadimLite"),
    "PatchCore": (".patchcore.patchcore", "PatchCore"),
    "RD4AD": (".rd4ad.rd4ad", "RD4AD"),
    "STFPM": (".stfpm.stfpm", "STFPM"),
    "Stfpm": (".paste.stfpm", "Stfpm"),
    "SuperSimpleNet": (".supersimplenet.supersimplenet", "SuperSimpleNet"),
    "VADModel": (".vad_model", "VADModel"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .cfa.cfa import CFA
    from .ganomaly.ganomaly import Ganomaly
    from .padim.padim import Padim
    from .padim.padim_lite import PadimLite
    from .paste.stfpm import Stfpm
    from .patchcore.patchcore import PatchCore
    from .rd4ad.rd4ad import RD4AD
    from .stfpm.stfpm import STFPM
    from .supersimplenet.supersimplenet import SuperSimpleNet
    from .vad_model import VADModel
