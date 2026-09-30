"""Characterization tests for ELM packing and humidity helpers (T-7).

These pin current behavior ahead of refactoring. Assertions marked
``PINS BUG Bn`` document known-incorrect behavior (see REFACTOR_NOTES.md)
and should be updated when that bug is fixed.
"""

import numpy as np

from dapper.elm.utils import (
    compute_humidities,
    compute_specific_humidity_from_rh,
    elm_var_packing_params,
)
from dapper.met.adapters.base import BaseAdapter


class _Adapter(BaseAdapter):
    def discover_files(self, csv_directory, calendar, *, clip_to_full_years=None):
        raise NotImplementedError

    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        raise NotImplementedError


def test_packing_params_from_preset_ranges():
    np.testing.assert_allclose(
        elm_var_packing_params("TBOT"), (262.50148352859395, 0.0029670571879079704)
    )
    # QBOT preset range is [0, 0.1] here (differs from schemas.elm.ELM_RANGES).
    np.testing.assert_allclose(
        elm_var_packing_params("QBOT"), (0.050000847730625124, 1.695461250233126e-06)
    )
    np.testing.assert_allclose(
        elm_var_packing_params("PRECTmms"),
        (6.781845000927711e-07, 1.3563690001865008e-06),
    )


def test_packing_params_from_data_and_dtype():
    data = np.array([250.0, 300.0])
    np.testing.assert_allclose(
        elm_var_packing_params("TBOT", data=data),
        (275.00042386531254, 0.000847730625116563),
    )
    np.testing.assert_allclose(
        elm_var_packing_params("TBOT", data=data, dtype=np.int32),
        (275.00000000646753, 1.2935035763233137e-08),
    )


def test_packing_params_with_nan_data():
    # PINS BUG B3: NaNs propagate into offset/scale instead of being ignored.
    ao, sf = elm_var_packing_params("QBOT", data=np.array([1e-3, np.nan, 3e-3]))
    assert np.isnan(ao) and np.isnan(sf)


def test_base_adapter_pack_params_fallbacks():
    adapter = _Adapter()
    np.testing.assert_allclose(
        adapter.pack_params("TBOT"), (262.50148352859395, 0.0029670571879079704)
    )
    assert adapter.pack_params("UNKNOWN") == (0.0, 1.0)
    np.testing.assert_allclose(
        adapter.pack_params("UNKNOWN", data=np.array([3.0, 5.0])),
        (4.000016954612502, 3.3909225004662515e-05),
    )


def test_compute_humidities_reference_values():
    rh, q = compute_humidities(
        np.array([290.0, 260.0, 280.0]),
        np.array([285.0, 255.0, 270.0]),
        np.array([1e5, 9e4, 1e5]),
    )
    # PINS BUG B2: vapor-pressure phase branches are inverted.
    np.testing.assert_allclose(
        rh, [82.31990210202056, 78.16320950686074, 47.77448854886398]
    )
    np.testing.assert_allclose(
        q, [0.00974753701782133, 0.0010576985363737, 0.0029276559349793]
    )


def test_compute_humidities_saturated_above_100_percent():
    # PINS BUG B2: Td == T should give RH == 100, but gives > 100.
    rh, _ = compute_humidities(
        np.array([290.0, 260.0]), np.array([290.0, 260.0]), np.array([1e5, 1e5])
    )
    np.testing.assert_allclose(rh, [119.41658607, 116.70298262])


def test_specific_humidity_from_rh():
    np.testing.assert_allclose(
        compute_specific_humidity_from_rh(
            np.array([20.0, -5.0]), np.array([50.0, 100.0]), np.array([1e5, 8e4])
        ),
        [0.00730014907167996, 0.00328753506404747],
    )
