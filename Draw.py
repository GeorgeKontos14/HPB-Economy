import torch
from Steps import *
import Variables.PreComputed as PreComputed
import Variables.Store as Store
import Variables.State as State

def initialize(regions: list[Region],
               no_kappas: int=25,
               no_lambdas: int=25,
               no_thetas: int=100):
    State.p_c_kappa = torch.ones(no_kappas)/no_kappas
    State.p_g_kappa = torch.ones(no_kappas)/no_kappas
    State.p_h_kappa = torch.ones(no_kappas)/no_kappas

    State.p_c_theta = torch.ones(no_thetas)/no_thetas
    State.p_g_theta = torch.ones(no_thetas)/no_thetas
    State.p_h_theta = torch.ones(no_thetas)/no_thetas

    State.p_c_lambda = torch.ones(no_lambdas)/no_lambdas
    State.p_g_lambda = torch.ones(no_lambdas)/no_lambdas

    State.F = torch.zeros(q+1)
    State.S_m = torch.zeros(q+1)
    State.X = torch.zeros((n, q+1))
    State.C = torch.zeros((n, q+1))
    for i, r in enumerate(regions):
        State.X[i] = torch.matmul(r.AApAi, r.Y)
        State.C[i] = State.X[i]-State.F
    State.Y0 = torch.zeros(q+1)

    State.sigma_m2 = 10**(-6)
    State.sigma_Da2 = 0.03**2/2.198
    State.ind_rho = 0
    State.mu_c = 0
    State.omega2 = 1
    State.kappa_c2 = torch.ones(n)
    State.kappa_g2 = torch.ones(25)
    State.kappa_h2 = torch.ones(10)
    State.lambda_c = torch.zeros(n)
    State.lambda_g = torch.zeros(25)
    State.ind_theta_c = torch.zeros(n).int()
    State.ind_theta_g = torch.zeros(25).int()
    State.ind_theta_h = torch.zeros(10).int()
    State.G = torch.zeros((25,q+1))
    State.H = torch.zeros((10, q+1))
    State.K = torch.zeros(25).int()
    State.J = torch.zeros(n).int()
    for i in range(25):
        State.ind_theta_g[i] = int(i%100)
        State.K[i] = int(i%10)
    for i in range(n):
        State.J[i] = int(i%25)
        State.ind_theta_c[i] = State.ind_theta_g[State.J[i]]
    for i in range(10):
        State.ind_theta_h[i] = int(i%100)

def draw():
    initialize(
        PreComputed.regions, 
        PreComputed.no_kappas, 
        PreComputed.no_lambdas, 
        PreComputed.no_thetas
    )

    step2(PreComputed.Sigma_U_inv)

    step3(PreComputed.Sigma_U_inv)

    step4(PreComputed.Sigma_U_inv,PreComputed.lambda_grid)

    step5(PreComputed.Sigma_U_inv,PreComputed.lambda_grid)

    step6(PreComputed.lambda_grid)

    step7(PreComputed.lambda_grid)

    step8(PreComputed.Sigma_U_inv, PreComputed.kappa_grid)

    step9(PreComputed.Sigma_U_inv, PreComputed.kappa_grid)

    step10(PreComputed.Sigma_U_inv, PreComputed.kappa_grid)

    step11(PreComputed.kappa_grid)

    step12(PreComputed.kappa_grid)

    step13(PreComputed.kappa_grid)

    step14(PreComputed.Sigma_U_inv)

    step15(PreComputed.Sigma_U_inv)

    step16(PreComputed.Sigma_U_inv, PreComputed.Det_Sigma_U)

    step17(PreComputed.Sigma_U_inv, PreComputed.Det_Sigma_U)

    step18(PreComputed.Sigma_U_inv, PreComputed.Det_Sigma_U)

    step19(PreComputed.no_thetas)

    step20(PreComputed.no_thetas)

    step21(PreComputed.no_thetas)

    step22(PreComputed.Sigma_U_inv)

    step23(PreComputed.Sigma_U_inv)

    step24(PreComputed.Sigma_m, PreComputed.Sigma_A)

    step25(
        PreComputed.Sigma_m, 
        PreComputed.Sigma_A, 
        PreComputed.Sigma_U_inv, 
        PreComputed.weights, 
        PreComputed.Deltainv
    )

    step26(PreComputed.Sigma_m, PreComputed.Sigma_A)

    step27(
        PreComputed.sigma_grid,
        PreComputed.Sigma_m_inv,
        PreComputed.Sigma_A_inv
    )

    step28(
        PreComputed.Sigma_m_inv, 
        PreComputed.Det_Sigma_m, 
        PreComputed.rho_grid
    )

def store_draw():
    Store.p_c_kappa_draws.append(State.p_c_kappa)
    Store.p_g_kappa_draws.append(State.p_g_kappa)
    Store.p_h_kappa_draws.append(State.p_h_kappa)

    Store.p_c_theta_draws.append(State.p_c_theta)
    Store.p_g_theta_draws.append(State.p_g_theta)
    Store.p_h_theta_draws.append(State.p_h_theta)

    Store.p_c_lambda_draws.append(State.p_c_lambda)
    Store.p_g_lambda_draws.append(State.p_g_lambda)

    Store.F_draws.append(State.F)
    Store.S_m_draws.append(State.S_m)
    Store.X_draws.append(State.X)
    Store.C_draws.append(State.C)

    Store.sigma_m2_draws.append(State.sigma_m2)
    Store.sigma_Da2_draws.append(State.sigma_Da2)
    Store.ind_rho_draws.append(State.ind_rho)
    Store.mu_c_draws.append(State.mu_c)
    Store.omega2_draws.append(State.omega2)
    Store.f0_draws.append(State.f0)
    Store.mu_m_draws.append(State.mu_m)

    Store.kappa_c2_draws.append(State.kappa_c2)
    Store.kappa_g2_draws.append(State.kappa_g2)
    Store.kappa_h2_draws.append(State.kappa_h2)

    Store.lambda_c_draws.append(State.lambda_c)
    Store.lambda_g_draws.append(State.lambda_g)

    Store.G_draws.append(State.G)
    Store.H_draws.append(State.H)
    Store.K_draws.append(State.K)
    Store.J_draws.append(State.J)

    Store.ind_theta_c_draws.append(State.ind_theta_c)
    Store.ind_theta_g_draws.append(State.ind_theta_g)
    Store.ind_theta_h_draws.append(State.ind_theta_h)

def sample(burn_in: int, total_draws: int):
    for _ in range(burn_in):
        draw()
    for _ in range(total_draws-burn_in):
        draw()
        store_draw()