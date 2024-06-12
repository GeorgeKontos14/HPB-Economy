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
            prob[j] = -0.5*torch.sum(torch.matmul(Sigma_U_inv[j],u)*u)/s2[i]+torch.log(p)
        prob = torch.exp(prob-torch.max(prob))
        for j in range(1,len(dist)):
            prob[j] += prob[j-1]
        prob = prob/prob[-1]
        ind = draw_proportional(prob)
        new_inds[i] = ind
    return new_inds.int()