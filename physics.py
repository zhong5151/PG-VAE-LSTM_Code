from __future__ import annotations

import numpy as np


def available_energy_from_net_radiation(net_radiation_wm2, ground_heat_wm2=None):
    """A_E = max(R_n - G, 0) in W m-2."""
    rn = np.asarray(net_radiation_wm2, dtype=np.float32)
    g = 0.0 if ground_heat_wm2 is None else np.asarray(ground_heat_wm2, dtype=np.float32)
    return np.maximum(rn - g, 0.0).astype(np.float32)


def available_energy_from_incoming_radiation(sw_down, lw_down, units="J_m2_day"):
    """Conservative incoming-radiation upper bound when net radiation is unavailable.

    This is looser than net available energy and should be described explicitly if used.
    """
    sw = np.asarray(sw_down, dtype=np.float32)
    lw = np.asarray(lw_down, dtype=np.float32)
    a = np.maximum(sw + lw, 0.0)
    if units == "J_m2_day":
        a = a / 86400.0
    elif units != "W_m2":
        raise ValueError("units must be 'J_m2_day' or 'W_m2'")
    return a.astype(np.float32)


def water_availability_mm_day(
    precipitation,
    soil_moisture_previous,
    soil_moisture_current,
    layer_depth_m=(0.07, 0.21),
    precip_units="m_day",
):
    """A_W = P + max(S_previous - S_current, 0), in mm day-1.

    Soil storage S is the sum of volumetric water content times layer thickness.
    For ERA5-Land layers 1 and 2, default thicknesses are 0.07 and 0.21 m.
    """
    p = np.asarray(precipitation, dtype=np.float32)
    if precip_units == "m_day":
        p = p * 1000.0
    elif precip_units != "mm_day":
        raise ValueError("precip_units must be 'm_day' or 'mm_day'")

    prev = np.asarray(soil_moisture_previous, dtype=np.float32)
    curr = np.asarray(soil_moisture_current, dtype=np.float32)
    depths = np.asarray(layer_depth_m, dtype=np.float32)
    if prev.shape[-1] != len(depths) or curr.shape[-1] != len(depths):
        raise ValueError("Last dimension of soil moisture arrays must match layer_depth_m")
    s_prev = np.sum(prev * depths * 1000.0, axis=-1)
    s_curr = np.sum(curr * depths * 1000.0, axis=-1)
    release = np.maximum(s_prev - s_curr, 0.0)
    return np.maximum(p, 0.0) + release
