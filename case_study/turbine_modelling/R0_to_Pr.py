import numpy as np

# 1) Your known data (rated power in kW, rotor radius in m)
P_r_known = np.array([2300, 4000, 5500, 7000])   # from 2.3, 4, 5.5, 7 MW
R0_known  = np.array([56.5,   75.0,  87.5, 100.0])

# 2) Fit log‐linear: log(R0) = n*log(P_r) + log(K)
logP = np.log(P_r_known)
logR = np.log(R0_known)
n, logK = np.polyfit(logP, logR, 1)
K = np.exp(logK)

print(f"Fitted power‐law: R0(P_r) = {K:.3f} * P_r**{n:.3f}")

# 3) Define the function
def R0_of_Pr(P_r: float) -> float:
    """
    Estimate the 'reference' rotor radius [m] for a turbine of rated power P_r [kW].
    """
    return K * (P_r ** n)

if __name__ == "__main__":

    # Example usage:
    for Pr in [2300, 4000, 5500, 7000]:
        print(f"P_r={Pr} kW → R0≈{R0_of_Pr(Pr):.1f} m")
