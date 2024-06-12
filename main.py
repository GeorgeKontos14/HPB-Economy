import time
from Prepare import *
from Steps import *

no_kappas: int=25
no_lambdas: int=25 
no_rhos: int=25
no_thetas: int=100 
no_sigmas: int=25
Deltavar: float = 0.01**2
pop_path = 'Data/pop_raw.csv'
yp_path = 'Data/yp_raw.csv'

def main():
    start = time.time()
    # Prepare data
    kappa_grid, lambda_grid, rho_grid, sigma_grid = grids(no_kappas, no_lambdas, no_rhos, no_sigmas)
    Xraw, R, Delta, Deltainv, Xfcstf, cutoff, G, V = baseline_trend(Deltavar)
    Sigma_m, Sigma_m_inv, Det_Sigma_m, Chol_Sigma_m, mfcstfm, cholfcstfm, ssv, ssh = Sigma_M(rho_grid, R)
    Sigma_A, Sigma_A_inv, Chol_Sigma_A, mfcstfa, cholfcstfa = Sigma_a(R)
    gammas, half_life_dist, theta = thetas(no_thetas)
    Sigma_U, Sigma_U_inv, Chol_Sigma_U, Det_Sigma_U, mfcstu, cholfcstu, Sfcstu = Sigma_Us(gammas,R,no_thetas)
    regions, F, SuAA, SuAAS, weights = loadRegions(no_thetas, pop_path,yp_path,R,V,cutoff,Xraw,Sigma_U)

    # Initialize Gibbs state
    initialize(regions, no_kappas, no_lambdas, no_thetas)

    # Gibbs Steps
    step1(Chol_Sigma_U,SuAA,SuAAS,weights,Delta)
    step2(Sigma_U_inv)
    step3(Sigma_U_inv)
    step4(Sigma_U_inv,lambda_grid)
    step5(Sigma_U_inv,lambda_grid)
    step6(lambda_grid)
    step7(lambda_grid)
    step8(Sigma_U_inv, kappa_grid)
    step9(Sigma_U_inv, kappa_grid)
    step10(Sigma_U_inv, kappa_grid)
    step11(kappa_grid)
    step12(kappa_grid)
    step13(kappa_grid)
    step14(Sigma_U_inv)
    step15(Sigma_U_inv)
    step16(Sigma_U_inv, Det_Sigma_U)
    step17(Sigma_U_inv, Det_Sigma_U)
    step18(Sigma_U_inv, Det_Sigma_U)
    step19(no_thetas)
    step20(no_thetas)
    step21(no_thetas)
    step22(Sigma_U_inv)
    step23(Sigma_U_inv)
    step24(Sigma_m, Sigma_A)
    step25(Sigma_m, Sigma_A, Sigma_U_inv, weights, Deltainv)
    step26(Sigma_m, Sigma_A)
    step27(sigma_grid,Sigma_m_inv,Sigma_A_inv)
    step28(Sigma_m_inv, Det_Sigma_m, rho_grid)

    end = time.time()
    print(f"Time elapsed: {end-start}")

if __name__ == "__main__":
    main()