import torch
import statsmodels.api as sm
import csv
import State
import PreComputed

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

T: int = 118 # Minimum number of years for a country
maxh: int = 100 # Offset between minimum and maximum number of years
Tmax: int = T+maxh # Maximum number of years for a country
n: int = 113 # Number of countries
q: int = 31
q0: int = 16
kmax: int = 800 # maximum AR(1) process half life

class Region:
    def __init__(self):
        self.qi = None
        self.Y = None
        self.w = None
        self.AApAi = None

def grids(no_kappas: int=25, 
          no_lambdas: int=25, 
          no_rhos: int=25,
          no_sigmas: int = 25):
    kappa_grid = torch.tensor([
        1/3*3**(2*i/(no_kappas-1)) for i in range(no_kappas)
    ])
    lambda_grid = torch.tensor([
        0.95*i/(no_lambdas-1) for i in range(no_lambdas)
    ])
    rho_grid = torch.tensor([
        0.5**(1/(50+i)*100/24) for i in range(no_rhos)
    ])
    
    sigma_grid = torch.tensor([
        0.01*(0.1+1.9*i/(no_sigmas-1)) for i in range(no_sigmas)
    ])
    return kappa_grid, lambda_grid, rho_grid, sigma_grid

def largest_eigenvecs(A: torch.Tensor, no_Vecs:int):
    eigvals, eigvecs = torch.linalg.eigh(A)
    indices = torch.argsort(eigvals, descending=True)[:no_Vecs]
    return eigvals[indices], eigvecs[:, indices]

def baseline_trend(Deltavar: float = 0.01**2):
    Xraw = torch.zeros((T,2))
    Xrawfcst = torch.zeros((Tmax-T+1, 2))
    V = torch.zeros((T,T))
    Xraw[:,0] = 1/T
    Xrawfcst[:,0] = 1/T
    Xraw[:,1] = torch.tensor([i/T for i in range(1,T+1)])
    Xrawfcst[:,1] = torch.tensor([i/T for i in range(T,Tmax+1)])
    Xrawfcst[:,1] = Xrawfcst[:,1]-torch.sum(Xrawfcst[:,1])/T
    Xraw[:,1] = Xraw[:,1]-torch.sum(Xraw[:,1])/T
    Xrawfcst = torch.matmul(Xrawfcst, 
                            torch.linalg.inv(torch.matmul(Xraw.t(), Xraw)))
    
    for i in range(T):
        for j in range(T):
            V[i][j] = min(i+1,j+1)
    M = torch.eye(T)-torch.linalg.multi_dot([
        Xraw, torch.linalg.inv(torch.matmul(Xraw.t(), Xraw)), Xraw.t()])
    evals, evecs = largest_eigenvecs(torch.linalg.multi_dot([M, V, M]), q-1)
    cutoff = 0.9999*evals[q0-1]
    R = torch.zeros((Tmax, q+101))
    Xfcstf = torch.zeros((100,2))
    R[:T,:2] = Xraw
    R[:T,2:(q+1)] = evecs
    R[T-1, (q+1):] = -1
    for s in range(100):
        R[T+s, q+1+s] = 1
        Xfcstf = Xrawfcst[s+1,:] - Xrawfcst[0, :]

    Delta = Deltavar*torch.matmul(R[:T, :q+1].t(), R[:T, :q+1])
    Deltainv = torch.linalg.inv(Delta)

    G = torch.matmul(R[:T,:(q+1)], torch.linalg.inv(torch.matmul(R[:T,:(q+1)].t(), R[:T,:(q+1)])))
    return Xraw, R, Delta, Deltainv, Xfcstf, cutoff, G, V

def Sigma_M(rho_grid: torch.Tensor, R: torch.Tensor):
    no_rhos = len(rho_grid)

    Sigma_m = torch.zeros((no_rhos, q+1, q+1))
    Sigma_m_inv = torch.zeros((no_rhos, q+1, q+1))
    Chol_Sigma_m = torch.zeros((no_rhos, q+1, q+1))
    Det_Sigma_m = torch.zeros(no_rhos)
    mfcstfm = torch.zeros(100,q+1,25)
    cholfcstfm = torch.zeros((100,100,25))
    ssv = torch.ones(Tmax)*(-1)
    ssh=torch.tensor([50])
    ssv[ssh+1]=1
    A = torch.zeros((Tmax, Tmax))

    for j in range(Tmax):
        for i in range(Tmax):
            A[i][j] = i>=j

    Sraw = torch.zeros((Tmax, Tmax))
    exps = torch.zeros((Tmax, Tmax))
    for s1 in range(Tmax):
        for s2 in range(Tmax):
            exps[s1][s2] = abs(s1-s2)

    for i, rho in enumerate(rho_grid):
        Sraw = rho**exps
        Sraw = torch.linalg.multi_dot([A, Sraw, A.t()])
        Sall = torch.linalg.multi_dot([R.t(), Sraw, R])
        S = Sall[:(q+1), :(q+1)]
        Det_Sigma_m[i] = torch.linalg.det(S)
        Sigma_m[i] = S
        Sigma_m_inv[i] = torch.linalg.inv(S)
        Chol_Sigma_m[i] = torch.linalg.cholesky(S)
        mfcstfm[:,:,i] = torch.matmul(Sall[(q+1):, :(q+1)], Sigma_m_inv[i,:,:])
        cholfcstfm[:,:,i] = torch.linalg.cholesky(Sall[(q+1):, (q+1):]-torch.matmul(
            mfcstfm[:,:,i], Sall[:(q+1),(q+1):]))

    return Sigma_m, Sigma_m_inv, Det_Sigma_m, Chol_Sigma_m, mfcstfm, cholfcstfm, ssv, ssh

def Sigma_a(R: torch.Tensor):
    A = torch.zeros((Tmax, Tmax))

    for j in range(Tmax):
        for i in range(Tmax):
            A[i][j] = i>=j

    Sraw = torch.matmul(A, A.t())
    Sall = torch.linalg.multi_dot([R.t(), Sraw, R])
    Sigma_A = Sall[:(q+1),:(q+1)]
    Chol_Sigma_A = torch.linalg.cholesky(Sigma_A)
    Sigma_A_inv = torch.linalg.inv(Sigma_A)
    mfcstfa = torch.matmul(Sall[(q+1):, :(q+1)], Sigma_A_inv)
    cholfcstfa=torch.linalg.cholesky(Sall[(q+1):, (q+1):]-torch.matmul(
        mfcstfa, Sall[:(q+1),(q+1):]))

    return Sigma_A, Sigma_A_inv, Chol_Sigma_A, mfcstfa, cholfcstfa

def thetas(no_thetas: int=100):
    gammas = torch.zeros((kmax+1, no_thetas))
    corr = torch.zeros(kmax+1,2)
    half_life_dist = torch.zeros(no_thetas)
    theta = torch.zeros((3,no_thetas))

    for i in range(no_thetas):
        x = torch.rand(3)
        hl = 25+775*x[:2]**2
        r = 2**(-1/hl)
        for j in range(kmax+1):
            corr[j] = r**j
        gammas[:,i] = torch.matmul(corr, torch.tensor([x[2],1-x[2]]))
        cond = gammas[1:, i] > 0.5
        half_life_dist[i] = cond.sum()
        theta[:,i] = torch.tensor([r[0],r[1],x[2]])

    return gammas, half_life_dist, theta

def setSfromga(ga):
    gax = torch.zeros(2*Tmax-1)
    gax[Tmax-1] = ga[0]
    S = torch.zeros(Tmax, Tmax)
    for k in range(1,Tmax):
        gax[k+Tmax-1] = ga[k]
        gax[Tmax-1-k] = ga[k]
    for k in range(Tmax):
        S[:,k] = gax[(Tmax-1-k):(2*Tmax-1-k)]
    return S

def Sigma_Us(gammas: torch.Tensor, R: torch.Tensor, no_thetas: int=100):
    Sigma_U = torch.zeros((no_thetas, q+1, q+1))
    Sigma_U_inv = torch.zeros((no_thetas, q+1, q+1))
    Chol_Sigma_U = torch.zeros((no_thetas, q+1, q+1))
    Det_Sigma_U = torch.zeros(no_thetas)
    mfcstu = torch.zeros((100,q+1,100))
    cholfcstu = torch.zeros((100,100,100))
    Sfcstu = torch.zeros((100,100,100))

    for i in range(no_thetas):
        Sraw = setSfromga(gammas[:,i])
        Sall = torch.linalg.multi_dot([R.t(), Sraw, R])
        S = Sall[:(q+1),:(q+1)]
        Sigma_U[i,:,:] = S
        Det_Sigma_U[i] = torch.linalg.det(S)
        Chol_Sigma_U[i,:,:] = torch.linalg.cholesky(S)
        Sigma_U_inv[i,:,:] = torch.linalg.inv(S)
        mfcstu[:,:,i] = torch.matmul(Sall[(q+1):, :(q+1)], Sigma_U_inv[i,:,:])
        Sfcstu[:,:,i] = Sall[(q+1):, (q+1):] - torch.matmul(
            mfcstu[:,:,i],Sall[:(q+1), (q+1):])
        cholfcstu[:,:,i] = torch.linalg.cholesky(Sfcstu[:,:,i])

    return Sigma_U, Sigma_U_inv, Chol_Sigma_U, Det_Sigma_U, mfcstu, cholfcstu, Sfcstu

def getlfweights(sel: torch.Tensor, 
                 V: torch.Tensor, 
                 cutoff: float, 
                 Xraw: torch.Tensor):
    X = torch.zeros((T,2))
    for i in range(2):
        X[:,i] = Xraw[:,i]*sel.float()
    M = torch.diag(sel.float())-torch.linalg.multi_dot([X, torch.linalg.inv(torch.matmul(X.t(),X)), X.t()])
    evals, evecs = largest_eigenvecs(torch.linalg.multi_dot([M, V, M]), q0-1)
    cond = evals>cutoff
    qw = cond.sum()
    w = torch.zeros((T, qw+2))
    w[:, :2] = X
    w[:, 2:] = evecs[:, :qw]
    return w

def setRegionwA(r: Region, 
                ind: int, 
                sel: torch.Tensor, 
                V: torch.Tensor, 
                cutoff: float, 
                Xraw: torch.Tensor, 
                R: torch.Tensor,
                Sigma_U: torch.Tensor,
                SuAA: torch.Tensor,
                SuAAS: torch.Tensor):
    notnans = []
    notnans.append(sel[0].item())
    for i in range(1,T-1):
        notnans.append(sel[i-1] or sel[i+1])
    notnans.append(sel[len(sel)-1].item())
    notnans = torch.tensor(notnans)
    r.w = getlfweights(notnans, V, cutoff, Xraw)
    r.qi = r.w.shape[1]-1
    AB = torch.zeros((q+1, q+1))
    R_n = R[:T, :(q+1)].numpy()
    for i in range(r.qi+1):
        w_n = r.w[:,i].numpy()
        model = sm.OLS(w_n, R_n, hasconst=False)
        results = model.fit()
        AB[i,:] = torch.tensor(results.params).float()
    AB[(r.qi+2):, :] = 0
    for i in range(r.qi+1, q+1):
        AB[i][i] = 1
    AB = AB.t()
    A = AB[:, :(r.qi+1)]
    r.AApAi = torch.matmul(A, torch.linalg.inv(torch.matmul(A.t(), A)))
    for i, Sigma in enumerate(Sigma_U):
        SuAA[:,:,i,ind] = torch.linalg.multi_dot([Sigma, A, torch.linalg.inv(torch.linalg.multi_dot([A.t(), Sigma, A])), A.t()])
    SuAAS[:,:,i,ind] = Sigma-torch.matmul(SuAA[:,:,i,ind], Sigma)

def loadRegions(no_thetas: int,
                pop_path: str,
                yp_path: str,
                R: torch.Tensor,
                V: torch.Tensor,
                cutoff: float,
                Xraw: torch.Tensor,
                Sigma_U: torch.Tensor):
    regions: list[Region] = [Region() for _ in range(n)]
    SuAA = torch.zeros((q+1, q+1, no_thetas, n))
    SuAAS = torch.zeros((q+1, q+1, no_thetas, n))

    mdata = torch.zeros((T, n))
    with open(pop_path, 'r') as file:
        rows = csv.reader(file)
        for i,row in enumerate(rows):
            for j, val in enumerate(row):
                mdata[i][j] = float(val)
    pop = torch.sum(mdata[65:75], dim=0)/10

    leveldata = torch.zeros((T, n))
    with open(yp_path, 'r') as file:
        rows = csv.reader(file)
        for i,row in enumerate(rows):
            for j, val in enumerate(row):
                leveldata[i][j] = float(val)
    leveldata=leveldata[(leveldata.shape[0]-T):, :]

    weights = pop/torch.sum(pop)
    F = torch.zeros(q+1)
    weff = 0
    for i in range(n):
        if weights[i] > 0 and not torch.isnan(leveldata[:,i]).any():
            F += weights[i]*torch.matmul(R[:T, :(q+1)].t(), torch.log(leveldata[:,i]))
            weff += weights[i]
    F = F/weff
    State.f0 = F[0]
    State.mu_m = F[1]
    for i, r in enumerate(regions):
        setRegionwA(r, i, ~torch.isnan(leveldata[:,i]), V, cutoff, Xraw, R, Sigma_U, SuAA, SuAAS)
        filtered = torch.where(~torch.isnan(leveldata[:,i]), leveldata[:,i], torch.tensor(1))
        r.Y = torch.matmul(r.w.t(), torch.log(filtered))
    
    return regions, F, SuAA, SuAAS, weights

def precompute(pop_path: str,
               yp_path: str):
    PreComputed.kappa_grid, PreComputed.lambda_grid, PreComputed.rho_grid, PreComputed.sigma_grid = grids(
        PreComputed.no_kappas,
        PreComputed.no_lambdas,
        PreComputed.no_rhos,
        PreComputed.no_sigmas
    )
    
    Xraw, R, PreComputed.Delta, PreComputed.Deltainv, Xfcstf, cutoff, G, V = baseline_trend(PreComputed.Deltavar)

    PreComputed.Sigma_m, PreComputed.Sigma_m_inv, PreComputed.Det_Sigma_m, Chol_Sigma_m, mfcstfm, cholfcstfm, ssv, ssh = Sigma_M(
        PreComputed.rho_grid, R)
    
    PreComputed.Sigma_A, PreComputed.Sigma_A_inv, Chol_Sigma_A, mfcstfa, cholfcstfa = Sigma_a(R)

    gammas, half_life_dist, theta = thetas(PreComputed.no_thetas)

    Sigma_U, PreComputed.Sigma_U_inv, PreComputed.Chol_Sigma_U, PreComputed.Det_Sigma_U,  mfcstu, cholfcstu, Sfcstu = Sigma_Us(
        gammas,R,PreComputed.no_thetas)
    
    PreComputed.regions, F, PreComputed.SuAA, PreComputed.SuAAS, PreComputed.weights = loadRegions(
        PreComputed.no_thetas, pop_path,yp_path,R,V,cutoff,Xraw,Sigma_U)