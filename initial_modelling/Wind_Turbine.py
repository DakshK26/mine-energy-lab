import numpy as np
import math as m


# Weidbull Distribution Function
def weibull_pdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    """
    Probability density function of the Weibull distribution.

    Parameters
    ----------
    k : shape parameter (k > 0)
    c : scale parameter (c > 0)
    v : wind speed array or scalar

    Returns
    -------
    PDF values for each wind speed in v
    """
    v = np.asarray(v) # Convert to numpy array
    return (k / c**k) * v**(k - 1) * np.exp(-(v / c)**k)


# Max value of wiedbull CDF
def weibull_cdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    """
    Cumulative distribution function of the Weibull distribution.

    Parameters
    ----------
    k : shape parameter (k > 0)
    c : scale parameter (c > 0)
    v : wind speed array or scalar

    Returns
    -------
    CDF values for each wind speed in v
    """
    v = np.asarray(v) # Convert to numpy array
    return 1 - np.exp(-(v / c)**k)

# Ideal Hub Height to Radius Relation from Literature
def calc_height(R: float) -> float:
    """
    Estimate hub height from rotor radius based on empirical relation.

    Parameters
    ----------
    R : rotor radius (m)

    Returns
    -------
    Predicted hub height (m)
    """
    return 2.7936 * (2 * R)**0.7663


# Power output at a specific speed
def power_output(
    rho0: float,
    R: float,
    Hm: float,
    Vu: np.ndarray,
    lam: float,
    theta: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    C5: float,
    C6: float,
    x: float
) -> np.ndarray:
    """
    Calculate mechanical power output of a wind turbine for given wind speeds.

    Parameters
    ----------
    rho0 : air density at reference level (kg/m^3)
    R    : rotor radius (m)
    Hm   : site elevation (m)
    Vu   : undisturbed wind speed array or scalar (m/s)
    lam  : tip speed ratio
    theta: blade pitch angle (rad)
    C1..C6, x : empirical coefficients

    Returns
    -------
    Power output array or scalar (W)
    """

    Vu = np.asarray(Vu)

    # Density correction for elevation
    density_corr = np.exp(-0.297 * Hm / 3048)

    # Rotor swept area
    A = np.pi * R**2

    # Power coefficient Cp(lambda, theta)
    cp_poly = C2 / lam - C3 * lam * theta - C4 * theta**x - C5
    cp = C1 * cp_poly * np.exp(-C6 / lam)

    # Mechanical power
    return 0.5 * rho0 * A * density_corr * Vu**3 * cp

# Calculate scale factor for weidbull distribution
def calc_scale(v_mean: float, k: float = 2.0) -> float:
    """
    Compute Weibull scale parameter from mean wind speed.

    v_mean : mean wind speed (m/s)
    k      : shape parameter (default 2.0)
    """
    return v_mean / m.gamma(1 + 1 / k)

# Mean turbine power (based on possibility of each power)
def mean_turbine_power(
    c: float,
    rho0: float,
    R: float,
    Hm: float,
    lam: float,
    theta: float,
    C1: float = 0.5,
    C2: float = 116,
    C3: float = 0.4,
    C4: float = 0.0,
    C5: float = 5.0,
    C6: float = 21.0,
    x: float = 0.0,
    k: float = 2.0,
    v_cut_in: float = 3.0,
    v_cut_out: float = 25.0,
    n: int = 1000
) -> float:
    """
    Estimate mean annual turbine power using the midpoint rule and the Weibull distribution.

    Parameters
    ----------
    c: scale factor
    rho0   : air density at reference (kg/m^3)
    R      : rotor radius (m)
    Hm     : site elevation (m)
    lam, theta : turbine parameters
    C1..C6, x  : Cp empirical coefficients
    k      : Weibull shape
    v_min, v_max : wind speed integration limits (m/s)
    n      : number of subintervals

    Returns
    -------
    Mean power (W)
    """

    # Calculate dv
    dv = (v_max - v_min) / n
    mid = v_min + (np.arange(n) + 0.5) * dv # Make array of midpoints

    # PDF and power at midpoints
    pdf = weibull_pdf(k, c, mid)
    p   = power_output(rho0, R, Hm, mid, lam, theta, C1, C2, C3, C4, C5, C6, x)

    # Sum multiplication of power and probability density function times dv
    return np.sum(p * pdf) * dv


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

# self testing main
if __name__ == "__main__":

    # Declare/initialize vars
    v_mean = 8.0       # m/s
    k = 2.0      # shape parameter
    c = calc_scale(v_mean, k)
    print(f"Scale parameter c: {c:.3f} m/s")

    # Test PDF and CDF at several wind speeds
    v_test = np.array([5.0, 8.0, 12.0])
    print("Weibull PDF:", weibull_pdf(k, c, v_test))
    print("Weibull CDF:", weibull_cdf(k, c, v_test))

    # Estimate hub height from rotor radius
    R  = 50.0  # m
    Hm = calc_height(R)
    print(f"Estimated hub height Hm: {Hm:.1f} m")

    # Power output for a range of wind speeds
    lam, theta = 7.0, 0.0
    cp_coeffs   = dict(C1=0.5, C2=116, C3=0.4, C4=0.0, C5=5.0, C6=21.0, x=0.0)
    P_out = power_output(
        rho0=1.225, R=R, Hm=Hm,
        Vu=v_test, lam=lam, theta=theta,
        **cp_coeffs
    )
    print("Power output [W]:", P_out)

    # Mean turbine power (midpoint rule)
    mean_P = mean_turbine_power(
        c, rho0=1.225, R=R, Hm=Hm,
        lam=lam, theta=theta, k=k, **cp_coeffs
    )
    print(f"Mean turbine power: {mean_P/1e3:.1f} kW")

    # Annual energy production (MWh/yr)
    AEP = mean_P * 8760 / 1e6
    print(f"Annual energy production: {AEP:.1f} MWh/yr")

    # Turbine cost estimate
    annual_cost = turbine_cost(
        P_r=mean_P/1e3, R=R, H=Hm,
        AEP=AEP, FCR=0.08
    )
    print(f"Annualized cost: ${annual_cost:,.0f}/yr")

