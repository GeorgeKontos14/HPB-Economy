"""
Variables for staring the results of each Gibbs draw
"""

import Utils.FileUtils as FileUtils

# Lists:
p_c_kappa_draws = []
p_g_kappa_draws = []
p_h_kappa_draws = []

p_c_theta_draws = []
p_g_theta_draws = []
p_h_theta_draws = []

p_c_lambda_draws = []
p_g_lambda_draws = []

F_draws = []
S_m_draws = []
X_draws = []
C_draws = []

sigma_m2_draws = []
sigma_Da2_draws = []
ind_rho_draws = []
mu_c_draws = []
omega2_draws = []
f0_draws = []
mu_m_draws = []

kappa_c2_draws = []
kappa_g2_draws = []
kappa_h2_draws = []

lambda_c_draws = []
lambda_g_draws = []

G_draws = []
H_draws = []
K_draws = []
J_draws = []

ind_theta_c_draws = []
ind_theta_g_draws = []
ind_theta_h_draws = []

# Paths to csv files:
paths = {
    "p_c_kappa_draws" : "Results/Distributions/p_c_kappa.csv",
    "p_g_kappa_draws" : "Results/Distributions/p_g_kappa.csv",
    "p_h_kappa_draws" : "Results/Distributions/p_h_kappa.csv",
    "p_c_theta_draws" : "Results/Distributions/p_c_theta.csv",
    "p_g_theta_draws" : "Results/Distributions/p_g_theta.csv",
    "p_h_theta_draws" : "Results/Distributions/p_h_theta.csv",
    "p_c_lambda_draws" : "Results/Distributions/p_c_lambda.csv",
    "p_g_lambda_draws" : "Results/Distributions/p_g_lambda.csv",
    "F_draws" : "Results/F.csv",
    "S_m_draws" : "Results/S_m.csv",
    "X_draws" : "Results/X.csv",
    "C_draws" : "Results/C.csv",
    "sigma_m2_draws" : "Results/sigma_m2.csv",
    "sigma_Da2_draws" : "Results/sigma_Da2.csv",
    "ind_rho_draws" : "Results/ind_rho.csv",
    "mu_c_draws" : "Results/mu_c.csv",
    "omega2_draws" : "Results/omega2.csv",
    "f0_draws" : "Results/f0.csv",
    "mu_m_draws" : "Results/mu_m.csv",
    "kappa_c2_draws" : "Results/Kappas/kappa_c2.csv",
    "kappa_g2_draws" : "Results/Kappas/kappa_g2.csv",
    "kappa_h2_draws" : "Results/Kappas/kappa_h2.csv",
    "lambda_c_draws" : "Results/Lambdas/lambda_c.csv",
    "lambda_g_draws" : "Results/Lambdas/lambda_g.csv",
    "G_draws" : "Results/G.csv",
    "H_draws" : "Results/H.csv",
    "K_draws" : "Results/K.csv",
    "J_draws" : "Results/J.csv",
    "ind_theta_c_draws" : "Results/Thetas/theta_c.csv",
    "ind_theta_g_draws" : "Results/Thetas/theta_g.csv",
    "ind_theta_h_draws" : "Results/Thetas/theta_h.csv"
}

def clear_files():
    for path in paths.values():
        FileUtils.clear_csv(path)

def write():
    for var, path in paths.items():
        FileUtils.write_to_file(globals()[var], path)
