import numpy as np
import math as m
from scipy.integrate import quad
import pandas as pd

'''
Configured for rated power from 2.3MW to 7MW. From radius of: 56.5 -> 100
'''
# m/s
V_CUT_IN = 3.25
V_RATED = 9.75
V_CUT_OUT = 25    
H0 = 82.9 # m

# Weibull Distribution Function
def weibull_pdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v) # Convert to numpy array
    return (k / c**k) * v**(k - 1) * np.exp(-(v / c)**k)

# Weibull CDF (array)
def weibull_cdf(k: float, c: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v) # Convert to numpy array
    return 1 - np.exp(-(v / c)**k)

# float
def weibull_cdf_scalar(k: float, c: float, v: float) -> float:
    """
    Weibull CDF for a single wind speed v (scalar).
    """
    return 1.0 - np.exp(- (v / c) ** k)


# Emperical Hub Height to Radius Relation from Literature
def calc_height(R: float) -> float:
    return 2.7936 * (2 * R)**0.7663

# Shear coefficient
def shear_coefficient(c0: float, h0: float):
    num = 0.37 - 0.0881 * m.log(c0)
    denom = 1-0.0881*m.log(h0/10)
    return num/denom

# Scale c as height changes
def compute_c(H: float, c0: float, H0: float, alpha: float):
    """
    Compute c according to:
        c = c0 * (H / H0) ** alpha
    Parameters:
    - H: Current value of H.
    - c0: Reference value of c.
    - H0: Reference value of H.
    - alpha: Wind Shear Factor.

    Returns:
    - Computed value of c.
    """
    return c0 * (H / H0) ** alpha

# Scale k as height changes
def compute_k(H, k0, H0):
    """
    Parameters:
    - H: Current value of H.
    - k0: Reference value of k.
    - H0: Reference value of H for k0.

    Returns:
    - Computed value of k.
    """
    numerator = 1 - 0.0881 * k0 * m.log(H0 / 10.0)
    denominator = 1 - 0.0881 * m.log(H / 10.0)
    return k0 * (numerator / denominator)

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

# Get power coefficient for rated power NREL
def calc_cp(P_r: float):
    C_p = (
        -7.677331e-04 * P_r**3
        + 1.022352e-02 * P_r**2
        - 4.313617e-02 * P_r
        + 4.568230e-01
    )
    
    return C_p

# Adjust for density at different heights
def density_adj(Hm: float):
    return m.exp(-0.297/3048 * Hm)

# Calculate power
def calc_power_mech(R: float, V: float, Cp: float, rho0: float = 1.225):
    '''
    R: m
    V: m/s
    rho0: kg/m^3
    
    Returns: Watts
    '''
    # Calculate area
    A = m.pi * R**2
    
    P_mech = 0.5 * rho0 * A * V**3 * Cp
    return P_mech
    
# Calculate power related to rated power at speeds for NREL turbines
def related_power_mech(P_r: float, V: float, a: float, b: float, c: float, d: float, g: float):
    '''
    P_r: kW
    V: m/s
    
    Returns: kW
    '''
    model = d + (a - d) / (1 + (V / c) ** b) ** g 
    P_v = P_r * model
    return P_v
    

# Calculate mean power per month
def monthly_power_output_elec(P_r: float, k: float, c_pdf: float, n_hours: float,
                              a: float, b: float, c: float, d: float, g: float,
                              Hm: float, n_drivetrain = 0.95):
    
    '''
    P_r: kW
    n_hours: hours
    Hm: m
    c_pdf: m/s
    
    Returns: kWh
    '''
    
    # Integrate from v cut in to v rated using related power function
    def integrand(v):
        # related_power_mech returns mechanical power [kW] at speed v
        return related_power_mech(P_r, v, a, b, c, d, g) * weibull_pdf(k, c_pdf, v)
    
    term1, _ = quad(integrand, V_CUT_IN, V_RATED)
    
    # Integrate from v rated to v cut out (can use CDF since P is a constant in this range)
    int = weibull_cdf_scalar(k, c_pdf, V_CUT_OUT) - weibull_cdf_scalar(k, c_pdf, V_RATED)
    term2 = P_r * int
    
    # Sum to get P_avg
    P_avg = term1 + term2
    
    # Multiply by drive train efficiencies to get electrical power from mech (and adjust air density for height)
    P_avg_elec = P_avg * n_drivetrain * density_adj(Hm)
    
    # Multiply by hours in month for monthly energy production
    MEP = P_avg_elec * n_hours
    
    return MEP
    


# Calculate yearly mean power
# Data Sets
def annual_energy_production(P_r: float,
                             Hm: float,
                             a: float,
                             b: float,
                             c_logistic: float,
                             d: float,
                             g: float,
                             n_drivetrain: float = 0.95) -> float:
    """
    Sum monthly electrical energy (kWh) over all 12 months,
    using hard‐coded (k, c) values per month.
    
    Parameters:
      P_r           : rated power [kW]
      a, b, c_logistic, d, g: 5PL logistic params
      rho0          : sea‐level air density [kg/m^3]
      n_drivetrain  : drivetrain efficiency
    
    Returns:
      Total annual energy (kWh)
    """
    
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

    # Days in each month (non‐leap year)
    _MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    _DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    total_annual = 0.0

    for month, days in zip(_MONTHS, _DAYS_IN_MONTH):
        # Look up k and c for this month
        k_month, c_month = _MONTHLY_WEIBULL[month]
        
        # Get shear coefficient
        alpha = shear_coefficient(c_month, H0)
        
        # Adjust k and c
        k_month = compute_k(Hm, k_month, H0)
        c_month = compute_c(Hm, c_month, H0, alpha)
        
        # Hours in this month
        hours = days * 24.0
        
        # Compute monthly energy (kWh)
        MEP = monthly_power_output_elec(
            P_r,
            k_month,
            c_month,
            hours,
            a, b, c_logistic, d, g,
            Hm,
            n_drivetrain
        )
        total_annual += MEP

    return total_annual

def costOfEnergy(cost, AEP):
    '''
    Cost: $ (USD)
    AEP: kWh
    '''
    return cost/AEP

if __name__ == "__main__":
    # Turbine specifications
    R_rotor = 175.0           # rotor diameter [m]
    H_hub   = 120.0           # hub height [m]
    P_r_kw  = 5500.0          # rated power [kW]

    # 5PL logistic coefficients (fitted):
    a = 0.037860093488330165
    b = 4.738520532230051
    c_log = 20.220591170903255
    d = 1.1999999999987006
    g = 49.99999994026665

    # 1) Compute hub height via calc_height, just to verify:
    R_radius = R_rotor / 2.0
    H_calc = calc_height(R_radius)
    print(f"Calculated hub height from R = {R_radius:.1f} m: "
          f"H = {H_calc:.2f} m  (Actual H_hub = {H_hub:.1f} m)")

    # 2) Compute Cp at rated power via polynomial:
    Cp_rated = calc_cp(P_r_kw/1000)
    print(f"Cp at P_r = {P_r_kw:.0f} kW: Cp = {Cp_rated:.4f}")

    # 3) Compute cost (just as a sanity check):
    #    First, estimate annual energy using placeholder Weibull k=2, c=8:
    test_k = 2.0
    test_c = 8.0
    sample_month_hours = 30 * 24.0
    sample_MEP = monthly_power_output_elec(
        P_r_kw, test_k, test_c, sample_month_hours,
        a, b, c_log, d, g, H_hub
    )
    # Rough annual via sample_MEP * 12:
    AEP_est = sample_MEP * 12.0
    cost_est = calculate_cost(R_radius, P_r_kw*1000, H_hub, AEP_est*1000)
    print(f"Estimated annual energy (using test k=2,c=8): {AEP_est:,.0f} kWh")
    print(f"Estimated cost (USD/yr) [using placeholder Weibull]: {cost_est:,.2f}")

    # 4) Compute full annual energy using actual monthly Weibull stats:
    AEP_actual = annual_energy_production(P_r_kw, H_calc, a, b, c_log, d, g)
    print(f"\nAnnual energy production (kWh) using monthly Weibull stats: {AEP_actual:,.0f} kWh")

    # 5) Now test the cost function with that actual AEP:
    cost_actual = calculate_cost(R_radius, P_r_kw*1000, H_hub, AEP_actual*1000)
    print(f"Cost (USD/yr) [using actual AEP]: {cost_actual:,.2f}")
    
    # Test density function
    print(f"Density adjusted for estimate height: {density_adj(H_calc)}")
    
    # Test v rated speed
    print(f"Power at V_rated: {calc_power_mech(R_radius, V_RATED, Cp_rated)}")
