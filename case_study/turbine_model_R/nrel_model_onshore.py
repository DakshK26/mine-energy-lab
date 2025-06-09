import os
import json
import numpy as np
import math as m
from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander

# ──────────────────────────────────────────────────────────────────────────────
# 1) Settings
# ──────────────────────────────────────────────────────────────────────────────
V_CUT_IN, V_RATED, V_CUT_OUT = 3.25, 9.75, 25.0
H0 = 82.9  # m (height where Weibull was originally measured)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Weibull functions
# ──────────────────────────────────────────────────────────────────────────────
def weibull_pdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return (k / c**k) * v**(k - 1) * np.exp(- (v / c)**k)

def weibull_cdf_scalar(k: float, c: float, v: float) -> float:
    return 1.0 - m.exp(- (v / c)**k)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Height & density adjustments
# ──────────────────────────────────────────────────────────────────────────────
def calc_height(R: float) -> float:
    """Empirical hub height from rotor radius R (m)."""
    return 2.7936 * (2 * R)**0.7663

def shear_coefficient(c0: float, h0: float) -> float:
    num   = 0.37 - 0.0881 * m.log(c0)
    denom = 1 - 0.0881 * m.log(h0 / 10.0)
    return num / denom

def compute_c(H: float, c0: float, H0: float, alpha: float) -> float:
    return c0 * (H / H0)**alpha

def compute_k(H: float, k0: float, H0: float) -> float:
    num   = 1 - 0.0881 * k0 * m.log(H0 / 10.0)
    denom = 1 - 0.0881 * m.log(H / 10.0)
    return k0 * (num / denom)

def density_adj(Hm: float) -> float:
    """Relative density at height Hm (m)."""
    return m.exp(-0.297 / 3048 * Hm)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Capital & O&M cost
# ──────────────────────────────────────────────────────────────────────────────
# Taken from Tables 1 & 2 of Section ...
def calculate_cost(R: float, P_r: float, H: float, AEP: float, FCR: float = 0.08):
    '''
    R: m
    H: m
    Rated Power: W
    AEP: Wh/yr
    '''
    # Initial Capital Costs
    # Mechanical system
    blade = ((0.4019 * R**3 - 955.24 + 2.7445 * R**2.5025) / 0.72) * 3  # 3 blades
    gearbox = 16.45 * (0.001 * P_r)**1.249
    low_speed_shaft = 0.1 * (2 * R)**2.887
    main_bearings = (0.64768 * R / 75 - 0.01068672) * (2 * R)**2.5
    mechanical_brake = 1.9894e-3 * P_r - 0.1141

    # Electrical system
    generator = 0.065 * P_r
    variable_speed_electronics = 0.079 * P_r
    electrical_connection = 0.04 * P_r
    
    # Control system
    pitch_system = 0.480168 * (2 * R)**2.6578
    yaw_system = 0.0678 * (2 * R)**2.964
    control_safety_system = 35000

    # Auxiliary system
    hydraulic_cooling = 0.012 * P_r
    hub = 2.0061666 * R**2.53 + 24141.275
    nose_cone = 206.69 * R - 2899.185
    mainframe = 11.9173875 * (2 * R)**1.953
    nacelle_cover = 1.1537e-2 * P_r + 3849.7
    tower = 0.59595 * m.pi * R**2 * H - 2121

    # Balance of station cost
    # Infrastructure
    foundation = 303.24 * (m.pi * R**2 * H)**0.4037
    roads_civil_work = 2.17e-15 * P_r**3 - 1.45e-8 * P_r**2 + 0.06954 * P_r
    electrical_interface = 3.49e-15 * P_r**3 - 2.21e-8 * P_r**2 + 0.1097 * P_r
    engineering_permits = 9.94e-10 * P_r**2 + 0.02031 * P_r

    # Installation and transportation
    transportation = 1.581e-14 * P_r**3 - 3.75e-8 * P_r**2 + 0.0547 * P_r
    installation = 1.965 * (2 * H * R)**1.1736
    
    # Initial Capital Cost
    ICC = (blade + gearbox + low_speed_shaft + main_bearings + mechanical_brake +
       generator + variable_speed_electronics + electrical_connection +
       pitch_system + yaw_system + control_safety_system +
       hydraulic_cooling + hub + nose_cone + mainframe + nacelle_cover + tower +
       foundation + roads_civil_work + electrical_interface + engineering_permits +
       transportation + installation)

    # Annual operating expenses
    replacement_cost = 0.00107 * P_r
    operations_maintenance = 7e-6 * AEP
    land_lease = 1.08e-6 * AEP

    AOE = replacement_cost + operations_maintenance + land_lease
    
    return ICC * FCR + AOE

# ──────────────────────────────────────────────────────────────────────────────
# 5) Chebyshev Cp model coefficients
# ──────────────────────────────────────────────────────────────────────────────
A = np.array([
    0.36331072494448835, 0.06160623375655268, -0.03009958339557637,
    0.013482428928696819, -0.004704316035470147, 0.002197646498117906,
    -0.0003455338815170328, 0.0005428578185082264, 0.00010442761449940066,
    0.0002496224987423643, 0.00010556279795830771, 0.0001881413004332004,
    -1.745850070781232e-05, 0.00017134050345880117, -5.112487778968863e-05,
    8.21783966297234e-05
])
B = np.array([
    -2.096235195183198e-07, -5.243480677826144e-07, 2.743813704020326e-09,
    -2.162152800494351e-07, -1.269416675423688e-07, -1.3989171999121358e-07,
    -1.0915000001965672e-07, -1.111219214061816e-07, -8.021360023337948e-08,
    -6.743229841509514e-08, -5.388045570906691e-08, -4.5268612277709276e-08,
    -1.691119038468164e-08, -2.8348158877376345e-08, -2.1200987363259594e-09,
    -6.725669579887082e-09
])

def _scale_to_cheb(v):
    """Map v in [V_CUT_IN, V_RATED] → x in [-1,1]."""
    return 2 * (v - V_CUT_IN) / (V_RATED - V_CUT_IN) - 1

def calc_cp(v, Pr):
    """
    Predict Cp for wind speed v [m/s] and rated power Pr [kW].
    Accepts scalar or array inputs.
    """
    v_arr, Pr_arr = np.asarray(v, float), np.asarray(Pr, float)
    v_flat, Pr_flat = v_arr.ravel(), Pr_arr.ravel()
    x = _scale_to_cheb(v_flat)
    T = chebvander(x, len(A)-1)            # shape (N,16)
    coefs = A[None,:] + B[None,:] * Pr_flat[:,None]
    cp_flat = np.sum(T * coefs, axis=1)
    return cp_flat.reshape(v_arr.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Mechanical power
# ──────────────────────────────────────────────────────────────────────────────
def calc_power_mech(R: float, V: float, Cp: float, rho0: float = 1.225) -> float:
    """Return mechanical power in Watts."""
    area = m.pi * R**2
    return 0.5 * rho0 * area * V**3 * Cp

# ──────────────────────────────────────────────────────────────────────────────
# 7) Monthly energy output
# ──────────────────────────────────────────────────────────────────────────────
def monthly_power_output_elec(P_r_kw: float,
                              R: float,
                              k: float,
                              c0: float,
                              n_hours: float,
                              Hm: float,
                              n_drivetrain: float = 0.95) -> float:
    """
    Compute monthly electrical energy (kWh) given Weibull(k,c0),
    rotor radius R, rated power P_r_kw, hours in month, hub height Hm.
    """
    def integrand(v):
        Cp   = calc_cp(v, P_r_kw)
        Pm_W = calc_power_mech(R, v, Cp)
        return (Pm_W / 1000.0) * weibull_pdf(k, c0, v)

    term1, _ = quad(integrand, V_CUT_IN, V_RATED)
    int2 = weibull_cdf_scalar(k, c0, V_CUT_OUT) - weibull_cdf_scalar(k, c0, V_RATED)
    term2 = P_r_kw * int2
    P_avg = term1 + term2
    P_elec = P_avg * n_drivetrain * density_adj(Hm)
    return P_elec * n_hours

# ──────────────────────────────────────────────────────────────────────────────
# 8) Annual energy production
# ──────────────────────────────────────────────────────────────────────────────
_MONTHLY_WEIBULL = {
    "January":   (1.730295873640355, 4.19357482410424),
    "February":  (1.5539947341238698, 4.263972641371435),
    "March":     (1.699590887545629,  4.613819397328386),
    "April":     (1.4231359644800003, 6.633386990000316),
    "May":       (2.0004889471765512, 7.464944817344972),
    "June":      (2.172663080432464,  7.089950124378428),
    "July":      (2.3155925071746375, 7.107493287002473),
    "August":    (2.5166786236232177, 7.985513540758083),
    "September": (1.7779669009470513, 6.793168891123515),
    "October":   (1.556819349544942,  5.522093396300041),
    "November":  (1.3779837098625127, 6.5706779432362685),
    "December":  (1.5031939051133054, 5.274959352934619)
}
_DAYS_IN_MONTH = {
    "January": 31, "February": 28, "March": 31, "April": 30,
    "May": 31, "June": 30, "July": 31, "August": 31,
    "September": 30, "October": 31, "November": 30, "December": 31
}

def annual_energy_production(P_r_kw: float,
                             R: float,
                             Hm: float,
                             n_drivetrain: float = 0.95) -> float:
    total = 0.0
    for month, (k0, c0) in _MONTHLY_WEIBULL.items():
        alpha = shear_coefficient(c0, H0)
        k_adj = compute_k(Hm, k0, H0)
        c_adj = compute_c(Hm, c0, H0, alpha)
        hours = _DAYS_IN_MONTH[month] * 24.0
        total += monthly_power_output_elec(
            P_r_kw, R, k_adj, c_adj,
            hours, Hm, n_drivetrain
        )
    return total

# Cost of energy function
def costOfEnergy(cost, AEP):
    '''
    $/kWh
    '''
    return cost/AEP

# ──────────────────────────────────────────────────────────────────────────────
# 9) Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Turbine spec
    R_rotor   = 175.0        # rotor diameter [m]
    R_radius  = R_rotor / 2  # rotor radius [m]
    H_hub     = 120.0        # hub height [m]
    P_r_kw    = 5500.0       # rated power [kW]

    # 1) Hub height from radius
    H_calc = calc_height(R_radius)
    print(f"Calculated H from R={R_radius:.1f} m → H={H_calc:.2f} m (actual {H_hub:.1f} m)")

    # 2) Cp at rated speed
    Cp_rated = calc_cp(V_RATED, P_r_kw)
    print(f"Cp @ V_RATED={V_RATED:.2f} m/s → Cp={Cp_rated:.4f}")

    # 3) Sanity‐check cost using placeholder Weibull
    test_k, test_c = 2.0, 8.0
    sample_hours   = 30 * 24.0
    sample_MEP     = monthly_power_output_elec(
        P_r_kw, R_radius, test_k, test_c,
        sample_hours, H_hub
    )
    AEP_est = sample_MEP * 12.0
    cost_est = calculate_cost(R_radius, P_r_kw*1000.0, H_hub, AEP_est*1000.0)
    print(f"Placeholder AEP (kWh/yr) = {AEP_est:,.0f}")
    print(f"Placeholder cost (USD/yr) = {cost_est:,.2f}")

    # 4) Full annual production
    AEP_actual = annual_energy_production(P_r_kw, R_radius, H_calc)
    print(f"\nAnnual energy production (kWh) = {AEP_actual:,.0f}")

    # 5) Cost with real AEP
    cost_actual = calculate_cost(R_radius, P_r_kw*1000.0, H_hub, AEP_actual*1000.0)
    print(f"Cost (USD/yr) = {cost_actual:,.2f}")

    # 6) Density at calculated height
    print(f"Density adj at H={H_calc:.1f} m = {density_adj(H_calc):.4f}")

    # 7) Power at rated speed
    P_rated_W = calc_power_mech(R_radius, V_RATED, Cp_rated)
    print(f"Mechanical power @ V_RATED = {P_rated_W/1e3:.2f} kW")
    
    # Cp at 5500 P_r and 9.75 rated
    print(f"Cp: {calc_cp(V_RATED, P_r_kw)}")
