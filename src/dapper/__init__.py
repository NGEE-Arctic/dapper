"""dapper public package interface.

The supported public import surface is:

- :class:`dapper.domains.domain.Domain`
- :class:`dapper.met.adapters.era5.ERA5Adapter`
- :class:`dapper.met.exporter.Exporter`
- :func:`dapper.integrations.era5.sample_era5_land`
- :func:`dapper.integrations.earthengine.gee_utils.sample_e5lh`

Other submodules may be importable, but are not considered part of the stable
"from dapper import ..." API.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "Domain",
    "ERA5Adapter",
    "ERA5SamplingPlan",
    "Exporter",
    "plan_era5_land_sampling",
    "sample_era5_land",
    "sample_e5lh",
]

if TYPE_CHECKING:  # pragma: no cover
    from dapper.domains.domain import Domain
    from dapper.met.adapters.era5 import ERA5Adapter
    from dapper.met.exporter import Exporter
    from dapper.integrations.era5 import (
        ERA5SamplingPlan,
        plan_era5_land_sampling,
        sample_era5_land,
    )
    from dapper.integrations.earthengine.gee_utils import sample_e5lh

_LAZY = {
    "Domain": ("dapper.domains.domain", "Domain"),
    "ERA5Adapter": ("dapper.met.adapters.era5", "ERA5Adapter"),
    "ERA5SamplingPlan": ("dapper.integrations.era5", "ERA5SamplingPlan"),
    "Exporter": ("dapper.met.exporter", "Exporter"),
    "plan_era5_land_sampling": (
        "dapper.integrations.era5",
        "plan_era5_land_sampling",
    ),
    "sample_era5_land": ("dapper.integrations.era5", "sample_era5_land"),
    "sample_e5lh": ("dapper.integrations.earthengine.gee_utils", "sample_e5lh"),
}


def __getattr__(name: str):
    """Lazily import public API symbols.

    This keeps ``import dapper`` lightweight (in particular, it avoids forcing
    imports of optional/heavy dependencies unless the corresponding functionality
    is used).
    """
    try:
        mod_name, attr = _LAZY[name]
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    mod = import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY)))
