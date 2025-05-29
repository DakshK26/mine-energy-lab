import numpy as np
import math as m

# Weibull Distribution Function
def weibull_pdf(k: float, c: float, v: np.ndarray) -> np.ndarray:

    v = np.asarray(v) # Convert to numpy array
    return (k / c**k) * v**(k - 1) * np.exp(-(v / c)**k)

# Weibull CDF
def weibull_cdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v) # Convert to numpy array
    return 1 - np.exp(-(v / c)**k)

# Ideal Hub Height to Radius Relation from Literature
def calc_height(R: float) -> float:
    return 2.7936 * (2 * R)**0.7663

# Calculate Power Coefficient for all Speeds till cut in
def power_coefficient_calc(V: np.ndarray, P_r: float)->np.ndarray:
    P_r = P_r/1000
    
    C_p = (
        (3.625e-3  - 2.774e-5 * P_r) * V**3 +
        (-8.090e-2  + 5.060e-4 * P_r) * V**2 +
        (5.881e-1   - 3.010e-3 * P_r) * V +
        (-9.961e-1  + 5.691e-3 * P_r)
    )
    
    return C_p

def power_output_mechanical(
    rho0: float,
    R: float,
    Hm: float,
    Vu: np.ndarray,
    Cp: np.ndarray,
) -> np.ndarray:
    """
    Compute mechanical power

    Parameters
    ----------
    rho0 : float
        Air density at reference level (kg/m^3).
    R : float
        Rotor radius (m).
    Hm : float
        Site elevation (m).
    Vu : array-like or float
        Undisturbed wind speed(s) (m/s).
    Cp : array-like or float
        Power coefficient(s) corresponding to each wind speed.

    Returns
    -------
    P_mech : numpy.ndarray or float
        mechanical power output kW
    """
    Vu = np.asarray(Vu)
    Cp = np.asarray(Cp)

    # Density correction for elevation
    density_corr = np.exp(-0.297 * Hm / 3048.0)

    # Rotor swept area
    A = np.pi * R**2

    # Mechanical power from wind
    P_mech = 0.5 * rho0 * A * density_corr * Vu**3 * Cp/1000 # To make kw

    return P_mech


def mean_turbine_power_piecewise(
    k: float,
    c: float,
    rho0: float,
    R: float,
    Hm: float,
    P_r: float,
    v_cut_in: float = 3.0,
    v_cut_out: float = 25.0,
    n: int = 1000
) -> float:
    """
    Estimate mean annual electrical turbine power using a piecewise model:
    - Below the wind speed where power_output exceeds rated power, use power_output().
    - Above that point, use rated power P_r.
    
    Parameters
    ----------
    c : float
        Weibull scale parameter.
    rho0 : float
        Air density at reference level (kg/m^3).
    R : float
        Rotor radius (m).
    Hm : float
        Site elevation (m).
    P_r : float
        Rated electrical power (kW).
    v_cut_in : float
        Cut-in wind speed (m/s).
    v_cut_out : float
        Cut-out wind speed (m/s).
    n : int
        Number of integration sub-intervals.
    
    Returns
    -------
    mean_P : float
        Mean annual electrical power (kW).
    """
    # Midpoint integration setup
    dv = (v_cut_out - v_cut_in) / n
    speeds = v_cut_in + (np.arange(n) + 0.5) * dv
    
    # Weibull PDF at midpoints
    pdf = weibull_pdf(k, c, v=speeds)  # shape k can be parameterized if needed
    
    # Instantaneous electrical power
    cp_vals = power_coefficient_calc(speeds, P_r)  # cp_model expects P_r in KW
    P_mech = power_output_mechanical(rho0, R, Hm, speeds, cp_vals)
    
    # Piecewise: cap at rated power
    P_used = np.minimum(P_mech, P_r)
    
    # Mean power via numerical integration
    mean_P = np.sum(P_used * pdf) * dv
    return mean_P


# Cost of wind turbine. From tables 1 & 2 of https://www.sciencedirect.com/science/article/pii/S0306261918310201#section-cited-by
def turbine_cost(
    P_r: float,
    R: float,
    H: float,
    AEP: float,
    FCR: float = 0.08
) -> float:
    """
    Annualized cost (USD/yr) for a wind turbine.

    P_r : rated power (kW)
    R   : rotor radius (m)
    H   : hub height (m)
    AEP : annual energy production (MWh/yr)
    FCR : fixed charge rate (yr^-1)
    """

    # Area
    area = np.pi * R**2

    # Component cost formulas (from literature)
    C_blade    = (0.4019 * R**2 - 955.24 * R + 2.7445) / (0.723 * R**2.5025)
    C_gb       = 16.45 * P_r**1.249
    C_shaft    = 2.415 * R**2.887
    C_bearings = (0.64768 * R/75 + 0.01068672) * R**2.5
    C_brake    = 1.9894e-3 * P_r**0.1141

    C_gen      = 0.065  * P_r
    C_elec     = 0.079  * P_r
    C_conn     = 0.04   * P_r

    C_pitch    = 0.480168 * R**2.6578
    C_yaw      = 0.0678   * R**2.964
    C_control  = 35000

    C_aux      = 0.012  * P_r
    C_hub      = 2.0061666 * R**2.53 + 24141.275
    C_nose     = -206.69 * R + 2899.185
    C_mainframe= 11.9173875 * R**1.953
    C_nacelle  = 1.1537e-2 * P_r**2 + 3849.7

    C_tower    = 0.59595 * area * H

    C_foundation = 303.24 * area * H**0.4037
    C_roads      = 2.17e-15 * P_r**3 + 1.45e-8 * P_r - 0.06954 * R**2
    C_elec_iface = 3.49e-15 * P_r**3 + 2.21e-8 * P_r - 0.1097 * R**2
    C_permits    = 9.94e-10 * P_r**2 + 0.02031 * P_r
    C_transport  = 1.581e-14 * P_r**3 + 3.75e-8 * P_r - 0.0547 * R**2
    C_install    = 1.965  * H * R**1.1736

    ICC = sum([
        C_blade, C_gb, C_shaft, C_bearings, C_brake,
        C_gen, C_elec, C_conn,
        C_pitch, C_yaw, C_control,
        C_aux, C_hub, C_nose, C_mainframe, C_nacelle,
        C_tower,
        C_foundation, C_roads, C_elec_iface, C_permits, C_transport, C_install
    ])

    # Annual operating expenses
    C_repl  = P_r**0.00107
    C_OandM = 7e-6  * AEP
    C_lease = 1.08e-6 * AEP
    AOE     = C_repl + C_OandM + C_lease

    return FCR * ICC + AOE
