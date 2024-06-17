import torch

def draw_standard_normal(n: int):
    dist = torch.distributions.MultivariateNormal(torch.zeros(n), torch.eye(n))
    return dist.sample()

def draw_proportional(prob: torch.tensor):
    unif = torch.rand(1)
    cond = unif>prob
    return min(cond.sum(),len(prob)-1)

def draw_index(meas: torch.Tensor,
               dist: torch.Tensor,
               s2: torch.Tensor,
               Sigma_U_inv: torch.Tensor,
               Det_Sigma_U: torch.Tensor):
    prob = torch.zeros(len(dist))
    new_inds = torch.zeros(len(meas))
    for i,u in enumerate(meas):
        for j, p in enumerate(dist):
            prob[j] = -0.5*torch.sum(torch.matmul(
                Sigma_U_inv[j],u)*u)/s2[i]+torch.log(p)+Det_Sigma_U[j]
        prob = torch.exp(prob-torch.max(prob))
        for j in range(1,len(dist)):
            prob[j] += prob[j-1]
        prob = prob/prob[-1]
        ind = draw_proportional(prob)
        new_inds[i] = ind
    return new_inds.int()

def decimal_representation(x: torch.Tensor):
    if x == 0:
        return 0
    sign = -1 if x < 0 else 1
    abs_x = torch.abs(x)
    n = torch.floor(torch.log10(abs_x))
    m = abs_x / 10 ** n
    a = sign*m
    b = n
    return a,b

def det(A: torch.Tensor):
    n = A.shape[0]
    base = torch.zeros(n)
    exp = torch.zeros(n)
    eigvals = torch.linalg.eigvals(A).real
    for i, val in enumerate(eigvals):
        a, b = decimal_representation(val)
        base[i] = a
        exp[i] = b
    base = torch.prod(base)
    exp = torch.sum(exp)
    base_a, base_b = decimal_representation(base)
    base = base_a
    exp += base_b
    return base, exp