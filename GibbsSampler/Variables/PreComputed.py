import torch

"""
Variables that are consistent throughout every run and
are computed deterministically
"""

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

Deltavar = 0.01**2
Delta = None
Deltainv = None

no_kappas = 25
no_lambdas = 25 
no_rhos = 25
no_thetas = 100 
no_sigmas = 25

lambda_grid = None
kappa_grid = None
rho_grid = None
sigma_grid = None

weights = None
regions = []

Sigma_U_inv = None
Chol_Sigma_U = None
Det_Sigma_U = None

SuAA = None
SuAAS = None

Sigma_m = None
Sigma_m_inv = None
Det_Sigma_m = None

Sigma_A = None
Sigma_A_inv = None