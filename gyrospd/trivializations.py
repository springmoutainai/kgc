# Code taken from the repository https://github.com/Lezcano/expRNN of one of 
# the authors of the paper 'Trivializations for Gradient-Based Optimization on Manifolds'
import torch

from gyrospd.expm32 import expm32, differential

def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]

class expm_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return expm32(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return differential(expm32, A.t(), G)

expm = expm_class.apply
