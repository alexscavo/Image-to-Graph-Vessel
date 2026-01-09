import copy
import torch
import torch.nn as nn

class EMA_Model(nn.Module):
    """
    Exponential Moving Average (EMA) model wrapper for a nn.Module.
    Keeps a shadow copy of the model parameters that is updated after optimizer steps.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.requires_grad_(False)  # EMA model does not require gradients
        self.ema.eval()             # Deterministic if there is not BN/dropout
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with the current model parameters."""
        
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        
        for k, v in model_state.items():
            
            # v = current EMA parameter
            # model_state[k] = current parameter from the model
            # alpha = decay
            
            # copy only floating point tensors
            if v.dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                ema_state[k].copy_(model_state[k])
                
    def to(self, device):
        """Move EMA model to the specified device."""
        self.ema.to(device)
        return self