import torch

def layer_init(layer, w_scale=1.0):
    torch.nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    torch.nn.init.constant_(layer.bias.data, 0.)
    return layer

def get_dist_params(output):
    mu = output[:,:,:,0]
    sigma = torch.nn.functional.softplus(output[:,:,:,1])
    return mu, sigma

def sample_from_output(output):
    if output.shape[-1] > 1:
        mu, sigma = get_dist_params(output)
        return torch.normal(mu, sigma)
    return output.squeeze(-1)