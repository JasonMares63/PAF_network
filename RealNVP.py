
import torch
import torch.nn as nn
from typing import Tuple
import math

class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dim,num_layers,
                dropout = 0.1):
        super().__init__()
        self.register_buffer("mask", mask)
        mask_features = mask.sum().int().item()
        in_features = mask_features 
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers_s, layers_t = [], []
        prev = in_features
        for _ in range(num_layers - 1):
            layers_s += [nn.Linear(prev, self.hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(self.dropout)]
            layers_t += [nn.Linear(prev, self.hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(self.dropout)]
            prev = self.hidden_dim
        layers_s.append(nn.Linear(prev, dim - mask_features))
        layers_s.append(nn.Tanh())
        layers_t.append(nn.Linear(prev, dim - mask_features))

        self.scale_net = nn.Sequential(*layers_s)
        self.translate_net = nn.Sequential(*layers_t)

    def forward(self, x):
        x1 = x[:, self.mask.bool()]
        x2 = x[:, ~self.mask.bool()]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        z2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        z = x.clone()
        z[:, ~self.mask.bool()] = z2
        return z, log_det

    def inverse(self, z):
        z1 = z[:, self.mask.bool()]
        z2 = z[:, ~self.mask.bool()]
        
        s = self.scale_net(z1)
        t = self.translate_net(z1)

        x2 = (z2 - t) * torch.exp(-s)

        x = z.clone()
        x[:, ~self.mask.bool()] = x2
        return x
    
class ActNorm(nn.Module):
    """
    Activation normalization layer (Glow-style).
    Data-driven initialization: first batch sets mean/scale to normalize.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor):
        with torch.no_grad():
            self.loc.data = -x.mean(0)
            self.scale.data = 1.0 / (x.std(0) + 1e-6)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.initialize(x)
        z = self.scale * (x + self.loc)
        log_det = self.scale.abs().log().sum().expand(x.shape[0])
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.scale - self.loc
 
class FactoredPrior(nn.Module):
    def __init__(self,  latent_dim=64):
        super().__init__()
        self.global_mean = nn.Parameter(torch.zeros(latent_dim))
        self.global_logvar = nn.Parameter(torch.zeros(latent_dim))
    
    def log_prob(self, z):
        z = z.clamp(min = 1e-8)
        mean   = self.global_mean
        logvar = self.global_logvar.clamp(-10,10)
        log_z = torch.log(z)
        log_LogNormal =  -0.5 * (
            (log_z - mean).pow(2) / logvar.exp()
            + logvar + math.log(2 * math.pi)) - log_z
        log_prob = log_LogNormal.sum(dim=-1)
        return log_prob
    
    def sample(self,temperature = 1.0):
        mean   = self.global_mean
        logvar = self.global_logvar.clamp(-10,10)
        std  = (0.5 * logvar).exp().clamp(min=0.1)
        z_gaussian = mean + std * torch.randn_like(mean) * temperature
        z_lognormal = torch.exp(z_gaussian)
        return(z_lognormal)
   
class RealNVP(nn.Module):
    """
    RealNVP normalizing flow 

    Learns p(x) as a pushforward of a LogNormal through
    a sequence of invertible transformations:
    """

    def __init__(
        self,
        dim: int, # n features of embeddings matrix
        num_layers : int = 3,
        n_flows: int = 8,
        hidden_dim: int = 128,
        dropout : int = 0.1,
        use_actnorm: bool = True,
        max_flows : int = 20,
        seed : int = 42
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_flows = n_flows
        self.max_flows = max_flows
        self.seed = seed
        self.dropout = dropout
        self.flows = nn.ModuleList()
        self.actnorms = nn.ModuleList() if use_actnorm else None
        
        torch.manual_seed(self.seed)   # reproducible masks across runs
        self.masks = torch.zeros(max_flows, self.dim)
        for i in range(max_flows):
            # Random mask — each flow layer gets a different random split
            perm      = torch.randperm(self.dim)
            mask      = torch.zeros(self.dim)
            mask[perm[:self.dim // 2]] = 1   # random half fixed, other half transformed
            self.masks[[i]] = mask

        for i in range(n_flows):
            self.flows.append(CouplingLayer(self.dim, self.masks[i], self.hidden_dim,
                                            self.num_layers,self.dropout))
            if use_actnorm:
                self.actnorms.append(ActNorm(self.dim))
                
        self.prior = FactoredPrior(
            self.dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode: x (data space) → z (latent space)
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for i, flow in enumerate(self.flows):
            if self.actnorms is not None:
                z, ld = self.actnorms[i](z)
                total_log_det += ld
            z, ld = flow(z)
            total_log_det += ld

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode: z (latent space) → x (data space)
        Used for sampling and in silico perturbations.
        """
        x = z
        for i in reversed(range(self.n_flows)):
            x = self.flows[i].inverse(x)
            if self.actnorms is not None:
                x = self.actnorms[i].inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return log p(z) + log determinant."""
        z, log_det = self.forward(x)
        log_pz = self.prior.log_prob(z)
        return -(log_pz + log_det)
    
    @torch.no_grad
    def _get_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under the learned distribution."""
        z, _ = self.forward(x)
        log_pz = self.prior.log_prob(z)
        return torch.exp(log_pz)

