import copy
import torch
import torch.nn as nn

class EMA_Model(nn.Module):
    """
    Exponential Moving Average (EMA) model wrapper for a nn.Module.
    Keeps a shadow copy of the model parameters that is updated after optimizer steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__() 
        self.decay = float(decay)

        # Shadow model
        self.ema = copy.deepcopy(model)
        self.ema.eval()  # deterministic if there is no BN/dropout

        # EMA params should never get grads
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with the current model parameters."""
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, v in model_state.items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                ema_state[k].copy_(v)

    def to(self, device):
        """Move EMA model to the specified device."""
        self.ema.to(device)
        return self