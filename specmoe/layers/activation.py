import torch
import torch.nn as nn
import torch.nn.functional as F
from sgl_kernel import silu_and_mul

class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return self.forward_cuda(x)
        else:
            return self.forward_native(x)
    
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        return out