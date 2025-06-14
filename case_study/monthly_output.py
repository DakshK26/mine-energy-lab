import os
import json
import numpy as np
import matplotlib.pyplot as plt
from turbine_modelling import nrel_model_onshore as turbine

class TurbineDetails():
    def __init__(self, R: float, P_r: float, H: float, n_drivetrain: float):
        '''
        R: m
        H: m
        P_r: kW
        n_drivetrain: --
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
    
    def calc_capacity_factor(self, AEP: float):
        return AEP/(self.P_r * 8760)
    
    # Method to decide mine energy consumption. Based on:
    # https://arctic-council.org/about/working-groups/acap/home/projects/arctic-black-carbon-case-studies-platform/diavik-wind-farm-yellow-knife-canada
    def mine_consumption(self, diesel_used: float = 50, diesel_prod: float = 10.6, generator_eff = 0.39):
        '''
        diesel_used: ML (mega litre)
        wind_power: GWh
        solar_power: GWh
        diesel_prod: 10.6 GWh/ML (https://www.dpi.nsw.gov.au/__data/assets/pdf_file/0011/665660/comparing-running-costs-of-diesel-lpg-and-electrical-pumpsets.pdf)
        generator_eff: 39% (https://www.sciencedirect.com/science/article/pii/S0306261923004002?) (average diesel genset efficiency in mine)
        ---
        returns: GWh/yr
        '''
        energy_produced = (diesel_used * diesel_prod * generator_eff) # convert to gwh
        return energy_produced

    def turbines_needed(self, mine_consumption: float, AEP: float):
        '''
        mine_consumption: GWh/yr
        AEP: kWh/yr
        ---
        Returns: 
        n: # of turbines needed
        Annual Energy Production: GWh/yr
        '''
        n = 1
        output = lambda x, y: (x * y)*1e-6  # Convert to GWh
        while output(n, AEP) < mine_consumption:
            n += 1
        return n, output(n, AEP)
    
def get_parm(File_Name: str = "turbine_modelling/optimized_turbine.json"):
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
    R = round(R) # Since radius will be built to a whole number
    P_r = round(P_r) # Same with rated power
    Hm = round(Hm)
    
    tne = TurbineDetails(R, P_r, Hm, n_drive)
    monthly_kwh = tne.monthlyOutputs()
    
    # Calculate variables
    AEP = np.sum(monthly_kwh)
    ICC = turbine.calc_ICC(R, P_r*1000, Hm)
    Annual_Cost = turbine.calculate_cost(R, P_r*1000, Hm, AEP*1000)
    COE = turbine.costOfEnergy(Annual_Cost, AEP)
    CF = tne.calc_capacity_factor(AEP)
    energy_needed = tne.mine_consumption()
    n_turbines, AEP_adj = tne.turbines_needed(energy_needed, AEP)
    monthly_need = energy_needed/12 # Assume constant amount of energy needed per month
    
    # Print
    print(f"Ideal Parameters:\nRadius: {R}, Rated Power: {P_r}, Hub Height: {Hm}")
    print("For One Turbine:")
    print(f"Initial Capital Cost: ${ICC:,.2f}")
    print(f"Annual Cost: {Annual_Cost:,.2f} $/yr")
    print("Monthly energy outputs (kWh):")
    print(np.round(monthly_kwh, 2))
    print(f"Annual Output: {AEP:,.2f} kWh")
    print(f"Cost of Energy: {COE:.5f} $/kWh")
    print(f"Capacity Factor: {CF*100:.2f}%")
    print(f"")
    print(f"Energy needed in average remote mine: {energy_needed:.2f} GWh/yr")
    print(f"Initial Capital Cost: ${ICC*n_turbines:,.2f}")
    print(f"Annual Cost: {Annual_Cost*n_turbines:,.2f} $/yr")
    print(f"Turbines needed: {n_turbines}, Annual Energy Production: {AEP_adj:,.2f} GWh/yr")

    # Plot
    months = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]
    plt.figure(figsize=(10,6))
    plt.bar(months, monthly_kwh*(n_turbines)*1e-6, color="skyblue")
    plt.axhline(monthly_need, color='red', linestyle='--', label='Monthly Energy Demand')
    plt.xlabel("Month")
    plt.ylabel("Energy (GWh)")
    plt.title("Monthly Energy Output")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
