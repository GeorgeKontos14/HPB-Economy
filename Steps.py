import torch
import State
from Prepare import *

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
    State.ind_rho = 0
    State.mu_c = 0
    State.omega2 = 1
    State.kappa_c = torch.ones(n)
    State.kappa_g = torch.ones(25)
    State.kappa_h = torch.ones(10)
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
        State.K[i] = int(i%25)
    for i in range(n):
        State.J[i] = int(i%25)
        State.ind_theta_c[i] = State.ind_theta_g[State.J[i]]
    for i in range(10):
        State.ind_theta_h[i] = int(i%100)