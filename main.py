import time
from Prepare import *
from Steps import *

no_kappas: int=25
no_lambdas: int=25 
no_rhos: int=25
no_thetas: int=100 
Deltavar: float = 0.01**2
pop_path = 'Data/pop_raw.csv'
yp_path = 'Data/yp_raw.csv'

def main():
    start = time.time()
    # Prepare data
    kappa_grid, lambda_grid, rho_grid = grids(no_kappas, no_lambdas, no_rhos)
    Xraw, R, Delta, Deltainv, Xfcstf, cutoff, G, V = baseline_trend(Deltavar)
    Sigma_m, Sigma_m_inv, Det_Sigma_m, Chol_Sigma_m, mfcstfm, cholfcstfm, ssv, ssh = Sigma_M(rho_grid, R)
    Sigma_A, Sigma_A_inv, Chol_Sigma_A, mfcstfa, cholfcstfa = Sigma_a(R)
    gammas, half_life_dist, theta = thetas(no_thetas)
    Sigma_U, Sigma_U_inv, Chol_Sigma_U, Det_Sigma_U, mfcstu, cholfcstu, Sfcstu = Sigma_Us(gammas,R,no_thetas)
    regions, F, SuAA, SuAAS = loadRegions(no_thetas, pop_path,yp_path,R,V,cutoff,Xraw,Sigma_U)

    # Initialize Gibbs state
    initialize(regions, no_kappas, no_lambdas, no_thetas)

    end = time.time()
    print(f"Time elapsed: {end-start}")

if __name__ == "__main__":
    main()