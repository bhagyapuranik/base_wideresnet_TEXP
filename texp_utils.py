import torch
import torch.nn as nn
import numpy as np


class ImplicitNormalizationConv(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        inp_norm = False
        if inp_norm :
            divisor = torch.norm(x.reshape(x.shape[0], -1), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  
            x = x*np.sqrt(x.numel()/x.shape[0])/(divisor+1e-6)
        
        weight_norms = (self.weight**2).sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1).sqrt()

        conv = super().forward(x)
        return conv/(weight_norms+1e-6)
    

class TexpNormalization(nn.Module):
    r"""Applies tilted exponential normalization over an input signal composed of several input
    planes.

    Args:
        tilt: Tilt of the exponential function, must be > 0.


    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`


    """

    def __init__(
        self,  
        tilt: float = 1.0,
        texp_across_filts_only: bool = True
        ) -> None:
        super(TexpNormalization, self).__init__()

        self.tilt = tilt
        self.texp_across_filts_only = texp_across_filts_only


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns softmax of input tensor, for each image in batch"""
        if self.texp_across_filts_only:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1),keepdim=True)
        else:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1,2,3),keepdim=True)

    def __repr__(self) -> str:
        s = "TexpNormalization("
        s += f'tilt={self.tilt}_filts_only_{self.texp_across_filts_only}'
        s += ")"
        return s



class AdaptiveThreshold(nn.Module):
    r"""
    Thresholds values x[x>threshold]
    """

    def __init__(self, std_scalar: float = 0.0, mean_plus_std: bool=True) -> None:
        super(AdaptiveThreshold, self).__init__()

        self.std_scalar = std_scalar # misnomer, it is a scale for std.
        self.means_plus_std = mean_plus_std

    def _thresholding(self, x, threshold):
        return x*(x > threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means = input.mean(dim=(2, 3), keepdim=True)
        stds = input.std(dim=(2, 3), keepdim=True)
        if self.means_plus_std:
            return self._thresholding(input, means + self.std_scalar*stds)
        else:
            return self._thresholding(input, means)

    def __repr__(self) -> str:
        if self.means_plus_std:
            s = f"AdaptiveThreshold(mean_plus_std_std_scale={self.std_scalar})"
        return s


from typing import Dict, Iterable, Callable
from collections import OrderedDict

class SpecificLayerTypeOutputExtractor_wrapper(nn.Module):
    def __init__(self, model: nn.Module, layer_type = nn.Conv2d):
        super().__init__()
        self._model = model
        self.layer_type = layer_type

        
        self.hook_handles = {}

        self.layer_outputs = OrderedDict()
        self.layer_inputs = OrderedDict()
        self.layers_of_interest = OrderedDict()

        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type) and ('module.conv1.0' ==layer_id): #'layer1' in layer_id or 
                print(layer_id)
                self.layer_outputs[layer_id] = torch.empty(0)
                self.layer_inputs[layer_id] = torch.empty(0)
                self.layers_of_interest[layer_id] = layer

        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type)  and ( 'module.conv1.0' ==layer_id): #'layer1' in layer_id or
                self.hook_handles[layer_id] = layer.register_forward_hook(
                    self.generate_hook_fn(layer_id))
                    
    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.layer_outputs[layer_id] = output
            self.layer_inputs[layer_id] = input[0]
        return fn

    def close(self):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def forward(self, x):
        return self._model(x)

    def __getattribute__(self, name: str):
        # the last three are used in nn.Module.__setattr__
        if name in ["_model", "layers_of_interest", "layer_outputs", "layer_inputs", "hook_handles", "generate_hook_fn", "close", "__dict__", "_parameters", "_buffers", "_non_persistent_buffers_set", "call_super_init"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self._model, name)
        

from optparse import Option
from typing import Dict, Union, Tuple, Optional
import torch.nn.functional as F
import math

def tilted_loss(activations: torch.Tensor, tilt: float,  
                relu_on=False, anti_hebb=False, texp_across_filts_only=True, patches: Optional[torch.Tensor] = None, weights:Optional[torch.Tensor] = None, **kwargs):
    if relu_on:
        activations = F.relu(activations)
    
    normalization = False # ALWAYS FALSE
    

    scaling_by_num_acts = True # Normalizes arguments of logsumexp by the number of activations, always true

    if anti_hebb:
        mean_subtraction = True # Subtracts the mean of the activations before exponentiating
    else:
        mean_subtraction = False

    if mean_subtraction:
        if texp_across_filts_only:
            mean_acts = torch.mean(activations, dim=(1), keepdim=True)
        else:
            mean_acts = torch.mean(activations, dim=(1,2,3), keepdim=True)
    else:
        mean_acts = torch.zeros_like(activations)

    # If no normalization is employed:
    if normalization == False:
        if scaling_by_num_acts:
            if texp_across_filts_only:
                return (1/(activations.shape[2]*activations.shape[3]))*(1/tilt)*torch.add( torch.sum(torch.logsumexp(tilt*(activations-mean_acts), dim=(1))), (activations.nelement()/activations.shape[1])*math.log(1/activations.shape[1]) )
            else:
                return (1/tilt)*torch.add( torch.sum(torch.logsumexp(tilt*(activations-mean_acts), dim=(1,2,3))), activations.shape[0]*math.log(activations.shape[0]/activations.nelement()) )
        else:
            if texp_across_filts_only:
                return (1/tilt)*torch.sum(torch.logsumexp(tilt*(activations - mean_acts), dim=(1)))
            return (1/tilt)*torch.sum(torch.logsumexp(tilt*(activations - mean_acts), dim=(1,2,3)))
    


