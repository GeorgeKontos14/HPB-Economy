import torch
import utils
import State
from Prepare import *

def step1(Chol_Sigma_U: torch.Tensor, 
          SuAA: torch.Tensor, 
          SuAAS: torch.Tensor,
          weights: torch.Tensor,
          Delta: torch.Tensor): # Draw X, C
    fhat = torch.zeros(q+1)
    sVs = torch.zeros((q+1,q+1))

    s2 = (1-State.lambda_c**2)*State.kappa_c2*State.omega2
    for ind, i in enumerate(State.J):
        mu_C = State.lambda_c[ind]*State.G[i]
        mu_C[0] += State.mu_c
        u = utils.draw_standard_normal(q+1)
        u = torch.sqrt(s2[ind])*torch.matmul(Chol_Sigma_U[State.ind_theta_c[ind]],
                                              u)+mu_C
        State.C[ind] = u - torch.matmul(SuAA[:,:,State.ind_theta_c[ind], ind],
                                         u-State.C[ind])
        if weights[ind] > 0:
            fhat += weights[ind]*State.C[ind]
            sVs += weights[ind]**2*s2[ind]*SuAAS[:,:,State.ind_theta_c[ind],ind]
    
    e = utils.draw_standard_normal(q+1)
    e = State.Y0+torch.matmul(torch.linalg.cholesky(Delta), e)
    fhat = torch.matmul(torch.linalg.inv(sVs+Delta), fhat-e)
    
    for i,w in enumerate(weights):
        if w > 0:
            State.C[i] -= w*s2[i]*torch.matmul(SuAAS[:,:,State.ind_theta_c[i],i], fhat)
        State.X[i] = State.C[i]-State.F

def step2(Sigma_U_inv: torch.Tensor): # G
    V_g = torch.zeros((25,q+1,q+1))
    ms = torch.zeros((25,q+1))

    s2 = (1-State.lambda_g**2)*State.kappa_g2*State.omega2
    for i,k in enumerate(State.K):
        V_g[i] = Sigma_U_inv[State.ind_theta_g[i]]/s2[i]
        ms[i] = torch.matmul(V_g[i], State.lambda_g[i]*State.H[k])
    
    s2 = (1-State.lambda_c**2)*State.kappa_c2*State.omega2
    for ind, i in enumerate(State.J):
        u = State.C[ind]
        u[0] -= State.mu_c
        V_g[i] += State.lambda_c[ind]*Sigma_U_inv[State.ind_theta_c[ind]]/s2[ind]
        ms[i] += State.lambda_c[ind]*torch.matmul(
            Sigma_U_inv[State.ind_theta_c[ind]],u)/s2[ind]
    
    for i,V in enumerate(V_g):
        inv_V = torch.linalg.inv(V)
        g = utils.draw_standard_normal(q+1)
        State.G[i] = torch.matmul(torch.linalg.cholesky(inv_V),
                                   g)+torch.matmul(inv_V, ms[i])

def step3(Sigma_U_inv): # H
    V_h = torch.zeros((10,q+1,q+1))
    ms = torch.zeros((10,q+1))

    s2 = State.kappa_h2*State.omega2
    for k in range(10):
        V_h[k] = Sigma_U_inv[State.ind_theta_h[k]]/s2[k]
    
    s2 = (1-State.lambda_g**2)*State.kappa_g2*State.omega2
    for i,k in enumerate(State.K):
        u = State.G[i]
        V_h[k] += State.lambda_g[i]*Sigma_U_inv[State.ind_theta_g[i]]/s2[i]
        ms[k] += State.lambda_g[i]*torch.matmul(
            Sigma_U_inv[State.ind_theta_g[i]],u)/s2[i]
        
    for k,V in enumerate(V_h):
        inv_V = torch.linalg.inv(V)
        h = utils.draw_standard_normal(q+1)
        State.H[k] = torch.matmul(torch.linalg.cholesky(inv_V),
                                  h)+torch.matmul(inv_V,ms[k])
        
def step4(Sigma_U_inv: torch.Tensor,
          lambda_grid: torch.Tensor): # lambda_c
    prob = torch.zeros(len(lambda_grid))
    for ind, i in enumerate(State.J):
        s2 = (1-lambda_grid**2)*State.kappa_c2[ind]*State.omega2
        for l, lam in enumerate(lambda_grid):
            u = State.C[ind]-lam*State.G[i]
            u[0] -= State.mu_c
            prob[l] = -0.5*torch.linalg.multi_dot([
                u.t(),Sigma_U_inv[State.ind_theta_c[ind]],
                u])/s2[l]-0.5*(q+1)*torch.log(s2[l])+torch.log(State.p_c_lambda[l])
        prob = torch.exp(prob-torch.max(prob))
        for l in range(1,25):
            prob[l] += prob[l-1]
        prob = prob/prob[-1]
        l_ind = utils.draw_proportional(prob)
        State.lambda_c[ind] = lambda_grid[l_ind]

def step5(Sigma_U_inv: torch.Tensor,
           lambda_grid: torch.Tensor): # lambda_g
    prob = torch.zeros(len(lambda_grid))
    for i,k in enumerate(State.K):
        s2 = (1-lambda_grid**2)*State.kappa_g2[i]*State.omega2
        for l, lam in enumerate(lambda_grid):
            u = State.G[i]-lam*State.H[k]
            prob[l] = -0.5*torch.linalg.multi_dot([
                u.t(),Sigma_U_inv[State.ind_theta_g[i]],
                u])/s2[l]-0.5*(q+1)*torch.log(s2[l])+torch.log(State.p_g_lambda[l])
        prob = torch.exp(prob-torch.max(prob))
        for l in range(1,len(lambda_grid)):
            prob[l] += prob[l-1]
        prob = prob/prob[-1]
        l_ind = utils.draw_proportional(prob)
        State.lambda_g[i] = lambda_grid[l_ind]

def step6(lambda_grid: torch.Tensor): # p_c_lambda
    no_lambdas = len(lambda_grid)
    a = torch.ones(no_lambdas)*20/no_lambdas
    for i in range(n):
        j = torch.round((no_lambdas-1)*State.lambda_c[i]/torch.max(lambda_grid)).int()
        a[j] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_c_lambda = prob/prob.sum()

def step7(lambda_grid: torch.Tensor): # p_g_lambda
    no_lambdas = len(lambda_grid)
    a = torch.ones(no_lambdas)*20/no_lambdas
    for i in range(25):
        j = torch.round((no_lambdas-1)*State.lambda_g[i]/torch.max(lambda_grid)).int()
        a[j] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_g_lambda = prob/prob.sum()

def step8(Sigma_U_inv: torch.Tensor,
          kappa_grid: torch.Tensor): # kappa_c
    prob = torch.zeros(len(kappa_grid))
    pbase = torch.zeros(len(kappa_grid))
    for l, kappa in enumerate(kappa_grid):
        pbase[l] = -0.5*(q+1)*torch.log(kappa)+torch.log(State.p_c_kappa[l])
    
    s2 = State.omega2*(1-State.lambda_c**2)
    for ind, i in enumerate(State.J):
        u = State.C[ind]-State.lambda_c[ind]*State.G[i]
        u[0] -= State.mu_c
        usu = torch.linalg.multi_dot([u.t(), Sigma_U_inv[State.ind_theta_c[ind]], u])
        prob = usu/(s2[ind]*kappa_grid)
        prob += pbase
        prob = torch.exp(prob-torch.max(prob))
        for l in range(1,len(kappa_grid)):
            prob[l] += prob[l-1]
        prob = prob/prob[-1]
        k_ind = utils.draw_proportional(prob)
        State.kappa_c2[ind] = kappa_grid[k_ind]
        
def step9(Sigma_U_inv: torch.Tensor,
          kappa_grid: torch.Tensor): # kappa_g
    prob = torch.zeros(len(kappa_grid))
    pbase = torch.zeros(len(kappa_grid))
    for l, kappa in enumerate(kappa_grid):
        pbase[l] = -0.5*(q+1)*torch.log(kappa)+torch.log(State.p_g_kappa[l])
    
    s2 = State.omega2*(1-State.lambda_g**2)
    for i,k in enumerate(State.K):
        u = State.G[i]-State.lambda_g[i]*State.H[k]
        usu = torch.linalg.multi_dot([u.t(), Sigma_U_inv[State.ind_theta_g[i]], u])
        prob = usu/(s2[i]*kappa_grid)
        prob += pbase
        prob = torch.exp(prob-torch.max(prob))
        for l in range(1,len(kappa_grid)):
            prob[l] += prob[l-1]
        prob = prob/prob[-1]
        k_ind = utils.draw_proportional(prob)
        State.kappa_g2[i] = kappa_grid[k_ind]

def step10(Sigma_U_inv: torch.Tensor,
          kappa_grid: torch.Tensor): # kappa_h
    prob = torch.zeros(len(kappa_grid))
    pbase = torch.zeros(len(kappa_grid))
    for l, kappa in enumerate(kappa_grid):
        pbase[l] = -0.5*(q+1)*torch.log(kappa)+torch.log(State.p_h_kappa[l])
    
    s2 = State.omega2
    for k, u in enumerate(State.H):
        usu = torch.linalg.multi_dot([u.t(), Sigma_U_inv[State.ind_theta_h[k]], u])
        prob = usu/(s2*kappa_grid)
        prob += pbase
        prob = torch.exp(prob-torch.max(prob))
        for l in range(1,len(kappa_grid)):
            prob[l] += prob[l-1]
        prob = prob/prob[-1]
        k_ind = utils.draw_proportional(prob)
        State.kappa_h2[k] = kappa_grid[k_ind]

def step11(kappa_grid: torch.Tensor): # p_c_kappa
    no_kappas = len(kappa_grid)
    a = torch.ones(no_kappas)*20/no_kappas
    for kappa in State.kappa_c2:
        cond = kappa >= kappa_grid
        ind = cond.sum()
        a[ind] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_c_kappa = prob/prob.sum()

def step12(kappa_grid: torch.Tensor): # p_g_kappa
    no_kappas = len(kappa_grid)
    a = torch.ones(no_kappas)*20/no_kappas
    for kappa in State.kappa_g2:
        cond = kappa >= kappa_grid
        ind = cond.sum()
        a[ind] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_g_kappa = prob/prob.sum()

def step13(kappa_grid: torch.Tensor): # p_h_kappa
    no_kappas = len(kappa_grid)
    a = torch.ones(no_kappas)*20/no_kappas
    for kappa in State.kappa_h2:
        cond = kappa >= kappa_grid
        ind = cond.sum()
        a[ind] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_h_kappa = prob/prob.sum()

def step14(Sigma_U_inv: torch.Tensor): # K
    prob = torch.zeros(10)
    s2 = (1-State.lambda_g**2)*State.kappa_g2*State.omega2
    for i,v in enumerate(State.G):
        for k,h in enumerate(State.H):
            u=v-State.lambda_g[i]*h
            prob[k] = -0.5*torch.linalg.multi_dot(
                [u.t(),Sigma_U_inv[State.ind_theta_g[i]],u])/s2[i]
        prob = torch.exp(prob-torch.max(prob))
        for k in range(1,10):
            prob[k] += prob[k-1]
        prob = prob/prob[-1]
        State.K[i] = utils.draw_proportional(prob)

def step15(Sigma_U_inv: torch.Tensor): # J
    prob = torch.zeros(25)
    s2 = (1-State.lambda_c**2)*State.kappa_c2*State.omega2
    for ind in range(n):
        v = State.C[ind]
        v[0] -= State.mu_c
        for k,g in enumerate(State.G):
            u=v-State.lambda_c[ind]*g
            prob[k] = -0.5*torch.linalg.multi_dot(
                [u.t(),Sigma_U_inv[State.ind_theta_c[ind]],u])/s2[ind]
        prob = torch.exp(prob-torch.max(prob))
        for i in range(1,25):
            prob[i] += prob[k-1]
        prob = prob/prob[-1]
        State.J[ind] = utils.draw_proportional(prob)

def step16(Sigma_U_inv: torch.Tensor,
           Det_Sigma_U: torch.Tensor): # ind_theta_c
    meas = torch.zeros((n,q+1))
    s2 = (1-State.lambda_c**2)*State.kappa_c2*State.omega2
    for ind,i in enumerate(State.J):
        meas[ind] = State.C[ind]-State.lambda_c[ind]*State.G[i]
        meas[ind][0] -= State.mu_c
    State.ind_theta_c = utils.draw_index(meas, State.p_c_theta, s2, Sigma_U_inv, Det_Sigma_U)

def step17(Sigma_U_inv: torch.Tensor,
           Det_Sigma_U: torch.Tensor): # ind_theta_g
    meas = torch.zeros((25,q+1))
    s2 = (1-State.lambda_g**2)*State.kappa_g2*State.omega2
    for i,k in enumerate(State.K):
        meas[i] = State.G[i]-State.lambda_g[i]*State.H[k]
    State.ind_theta_g = utils.draw_index(meas, State.p_g_theta, s2, Sigma_U_inv, Det_Sigma_U)

def step18(Sigma_U_inv: torch.Tensor,
           Det_Sigma_U: torch.Tensor): # ind_theta_h
    meas = torch.zeros((10,q+1))
    s2 = State.kappa_h2*State.omega2
    for i,h in enumerate(State.H):
        meas[i] = h
    State.ind_theta_h = utils.draw_index(meas, State.p_h_theta, s2, Sigma_U_inv, Det_Sigma_U)

def step19(no_thetas: int): # p_c_theta
    a = torch.ones(no_thetas)*20/no_thetas
    for x in State.ind_theta_c:
        a[x] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_c_theta = prob/prob.sum()

def step20(no_thetas: int): # p_g_theta
    a = torch.ones(no_thetas)*20/no_thetas
    for x in State.ind_theta_g:
        a[x] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_g_theta = prob/prob.sum()

def step21(no_thetas: int): # p_h_theta
    a = torch.ones(no_thetas)*20/no_thetas
    for x in State.ind_theta_h:
        a[x] += 1
    dist = torch.distributions.Chi2(a)
    prob = dist.sample()
    State.p_h_theta = prob/prob.sum()

def step22(Sigma_U_inv: torch.Tensor): # mu_c
    m = 0
    prec = 0
    s2 = State.omega2*(1-State.lambda_c**2)*State.kappa_c2
    i_1 = torch.zeros(q+1)
    i_1[0] = 1
    for ind, i in enumerate(State.J):
        u = State.C[ind]-State.lambda_c[ind]*State.G[i]
        m += torch.linalg.multi_dot(
            [i_1.t(), Sigma_U_inv[State.ind_theta_c[ind]], u])/s2[ind]
        prec += torch.linalg.multi_dot(
            [i_1.t(), Sigma_U_inv[State.ind_theta_c[ind]], i_1])/s2[ind]
    v = utils.draw_standard_normal(1)
    v = m/prec+v/torch.sqrt(prec)
    State.mu_c = v[0]

def step23(Sigma_U_inv: torch.Tensor):
    ssum = 1/2.198
    snu = 1
    s2 = State.kappa_c2*(1-State.lambda_c**2)
    for ind, i in enumerate(State.J):
        u = State.C[ind]-State.lambda_c[ind]*State.G[i]
        ssum += torch.linalg.multi_dot(
            [u.t(), Sigma_U_inv[State.ind_theta_c[ind]], u])/s2[ind]
        snu += q+1
    s2 = State.kappa_g2*(1-State.lambda_g**2)
    for i,k in enumerate(State.K):
        u = State.G[i]-State.lambda_g[i]*State.H[k]
        ssum += torch.linalg.multi_dot([
            u.t(), Sigma_U_inv[State.ind_theta_g[i]], u])/s2[i]
        snu += q+1
    s2 = State.kappa_h2
    for k, h in enumerate(State.H):
        ssum += torch.linalg.multi_dot([
            h.t(), Sigma_U_inv[State.ind_theta_h[k]], h])/s2[k]
        snu += q+1
    dist = torch.distributions.Chi2(snu)
    v = dist.sample()
    State.omega2 = ssum/v

def step24(Sigma_m: torch.Tensor,
           Sigma_A: torch.Tensor):
    Sigma_F = State.sigma_m2*Sigma_m[State.ind_rho]+State.sigma_Da2*Sigma_A
    Sigma_F_inv = torch.linalg.inv(Sigma_F)

    prec = Sigma_F_inv[:2,:2]
    m = torch.matmul(Sigma_F_inv[:2], State.F)
    v = utils.draw_standard_normal(2)
    v = torch.matmul(prec,m)+torch.matmul(
        torch.linalg.cholesky(prec), v)
    State.f0 = v[0]
    State.mu_m = v[1]

def step25(Sigma_m: torch.Tensor,
           Sigma_A: torch.Tensor,
           Sigma_U_inv: torch.Tensor,
           weights: torch.Tensor,
           Deltainv: torch.Tensor):
    m = torch.zeros(q+1)
    fhat = torch.zeros(q+1)
    Sigma_F = State.sigma_m2*Sigma_m[State.ind_rho]+State.sigma_Da2*Sigma_A

    s2 = State.omega2*State.kappa_c2*(1-State.lambda_c**2)
    for ind, i in enumerate(State.J):
        u = State.X[ind] - State.lambda_c[ind]*State.G[i]
        u[0] -= State.mu_c
        u[0] -= State.f0
        u[1] -= State.mu_m
        m += torch.matmul(Sigma_U_inv[State.ind_theta_c[ind]], u)/s2[ind]
        Sigma_F += Sigma_U_inv[State.ind_theta_c[ind]]/s2[ind]
        if weights[ind] > 0:
            fhat += weights[ind]*State.X[ind]
    Sigma_F_inv = torch.linalg.inv(Sigma_F)
    V_F = torch.linalg.inv(Sigma_F_inv+Deltainv)
    fhat[0] -= State.f0
    fhat[1] -= State.mu_m
    m += torch.matmul(Deltainv, fhat-State.Y0)
    v = utils.draw_standard_normal(q+1)
    State.F = torch.matmul(V_F, m)+torch.matmul(
        torch.linalg.cholesky(V_F), v)
    State.F[0] += State.f0
    State.F[1] += State.mu_m
    for ind in range(n):
        State.C[ind] = State.X[ind]-State.F

def step26(Sigma_m: torch.Tensor,
           Sigma_A: torch.Tensor): # S_m
    Sig_m = State.sigma_m2*Sigma_m[State.ind_rho]
    Sigma_S = Sig_m+State.sigma_Da2*Sigma_A
    u = State.F
    u[0] -= State.f0
    u[1] -= State.mu_m
    mfm = torch.matmul(Sig_m, torch.linalg.inv(Sigma_S))
    v = utils.draw_standard_normal(q+1)
    State.S_m = torch.matmul(mfm,u)+torch.matmul(
        torch.linalg.cholesky(Sig_m-torch.matmul(mfm,Sig_m)), v)
    
def step27(sigma_grid: torch.Tensor,
           Sigma_m_inv: torch.Tensor,
           Sigma_A_inv: torch.Tensor):
    no_sigmas = len(sigma_grid)
    prob = torch.zeros(no_sigmas)
    usu = torch.linalg.multi_dot(
        [State.S_m.t(), Sigma_m_inv[State.ind_rho], State.S_m])
    for l, sigma in enumerate(sigma_grid):
        prob[l] = -0.5*usu/sigma-0.5*(q+1)*torch.log(sigma)
    prob = torch.exp(prob-torch.max(prob))
    for l in range(no_sigmas):
        if 2*l <= no_sigmas:
            prob[l] = prob[l]*l
        else:
            prob[l] = prob[l]*(no_sigmas+1-l)
    for l in range(1, no_sigmas):
        prob[l] += prob[l-1]

    prob = prob/prob[-1]
    s_ind = utils.draw_proportional(prob)
    State.sigma_m2 = sigma_grid[s_ind]

    u = State.F-State.S_m
    u[0] -= State.f0
    u[1] -= State.mu_m
    ssum = 0.03**2/2.198+torch.linalg.multi_dot(
        [u.t(), Sigma_A_inv, u])
    snu = q+2
    dist = torch.distributions.Chi2(snu)
    v = dist.sample()
    State.sigma_Da2 = ssum/v

def step28(Sigma_m_inv: torch.Tensor,
           Det_Sigma_m: torch.Tensor,
           rho_grid: torch.Tensor): #ind_rho
    no_rhos = len(rho_grid)
    u = State.S_m
    prob = torch.zeros(no_rhos)
    for i in range(no_rhos):
        expon = -0.5/State.sigma_m2*torch.linalg.multi_dot([u.t(),Sigma_m_inv[i], u])
        prob[i] = torch.exp(expon) # /Det_Sigma_m[i]**2
    for i in range(1,no_rhos):
        prob[i] += prob[i-1]
    prob = prob/prob[-1]
    State.ind_rho = utils.draw_proportional(prob)