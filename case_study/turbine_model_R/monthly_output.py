import os
import json
import numpy as np
import matplotlib.pyplot as plt
import nrel_model_onshore as turbine

class TurbineDetails():
    def __init__(self, R: float, P_r: float, H: float, n_drivetrain: float):
        '''
        R: m
        H: m
        P_r: kW
        n_drivetrain: %
        '''
        self.R = R
        self.P_r = P_r
        self.H = H
        self.n_drivetrain = n_drivetrain

    def monthlyOutputs(self) -> np.ndarray:
        """
        Return a 12-element array of monthly electrical energy (kWh).
        """
        months = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]
        results = []
        for m in months:
            # raw Weibull at measurement height
            k0, c0 = turbine._MONTHLY_WEIBULL[m]
            days   = turbine._DAYS_IN_MONTH[m]
            hours  = days * 24.0

            # --- apply shear to k & c for hub height self.H ---
            alpha = turbine.shear_coefficient(c0, turbine.H0)
            k_adj = turbine.compute_k(self.H, k0, turbine.H0)
            c_adj = turbine.compute_c(self.H, c0, turbine.H0, alpha)

            MEP = turbine.monthly_power_output_elec(
                        self.P_r,         # kW
                        self.R,           # m
                        k_adj,            # adjusted k
                        c_adj,            # adjusted c
                        hours,
                        self.H,
                        self.n_drivetrain
                  )
            results.append(MEP)
        return np.array(results)
    
def get_parm(File_Name: str = "optimized_turbine.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path  = os.path.join(script_dir, File_Name)
    with open(json_path, "r") as jf:
        data = json.load(jf)

    R_opt   = data["Best Radius (m)"]
    P_r_opt = data["Best Rated Power (kW)"]
    H_opt   = data["Hub Height (m)"]
    n_drive = 0.95 # Default
    
    return R_opt, P_r_opt, H_opt, n_drive

if __name__ == "__main__":
    R, P_r, Hm, n_drive = get_parm()
    
    tne = TurbineDetails(R, P_r, Hm, n_drive)
    monthly_kwh = tne.monthlyOutputs()
    
    # Print details
    AEP = np.sum(monthly_kwh)
    ICC = turbine.calc_ICC(R, P_r*1000, Hm)
    Annual_Cost = turbine.calculate_cost(R, P_r*1000, Hm, AEP*1000)
    COE = turbine.costOfEnergy(Annual_Cost, AEP)
    print(f"Initial Capital Cost: {ICC}")
    print(f"Annual Cost: {Annual_Cost}")
    print("Monthly energy outputs (kWh):")
    print(monthly_kwh)
    print(f"Annual Output: {AEP} kWh")
    print(f"Cost of Energy: {COE}")

    # Plot
    months = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]
    plt.figure(figsize=(10,6))
    plt.bar(months, monthly_kwh, color="skyblue")
    plt.xlabel("Month")
    plt.ylabel("Energy (kWh)")
    plt.title("Monthly Energy Output")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
