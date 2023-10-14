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
    



def common_corruptions_test(x_orig, y_orig, device, model, verbose=False):    
    device = model.parameters().__next__().device

    model.eval()
    bs = 100
    n_batches = math.ceil(x_orig.shape[0] / bs)
    test_loss = 0
    test_correct = 0
    test_correct_top5 = 0

    for counter in range(n_batches):

        data = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        target = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target.type(torch.LongTensor).to(device)).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_correct_top5 += target.unsqueeze(dim=1).eq(torch.topk(output, k=5, dim=1)[1]).sum().item()

    test_size = x_orig.shape[0]

    if verbose:
        print(
            f"Corruption \t Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}, Test acc top5: {100*test_correct_top5/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size, test_correct_top5/test_size


from robustbench.data import  load_cifar100, load_cifar100c


def test_common_corruptions(device, model): 
    # Evaluate on common corruptions datatset:
    corruptions = ('gaussian_noise', 'shot_noise',  'impulse_noise','speckle_noise', 'snow', 'frost', 'fog', 'brightness',  'spatter','defocus_blur',  'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur','contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',  'saturate',)
    all_severities = True 
    test_acc_all = []
    test_acc_top5_all = []
    test_acc_sev5_all = []
    test_acc_sev5_top5_all = []

    for corr_idx in range(len(corruptions)):
        corruption = (corruptions[corr_idx],)
       
        x_test, y_test = load_cifar100c(n_examples=10000, data_dir='/home/bhagyap/datasets', corruptions=corruption, all_severities=all_severities)
        x_test_sev5, y_test_sev5 = load_cifar100c(n_examples=50000, data_dir='/home/bhagyap/datasets', corruptions=corruption)
    

        test_loss_corr, test_acc_corr, test_acc_top5_corr = common_corruptions_test(x_test, y_test, device, model)
        test_loss_corr_sev5, test_acc_corr_sev5, test_acc_top5_corr_sev5 = common_corruptions_test(x_test_sev5, y_test_sev5, device, model)

        test_acc_all.append(test_acc_corr)
        test_acc_sev5_all.append(test_acc_corr_sev5)

        test_acc_top5_all.append(test_acc_top5_corr)
        test_acc_sev5_top5_all.append(test_acc_top5_corr_sev5)
        
        print(f'Corruption: {corruption[0]} \t Test \t acc all: {test_acc_corr:.4f} \t acc sev5: {test_acc_corr_sev5:.4f} \t Top-5 acc all: {test_acc_top5_corr:.4f} \t Top-5 acc sev5: {test_acc_top5_corr_sev5:.4f}')
    
    print(f'Accuracy on all corruptions: {([val*100 for val in test_acc_all])}')
    print(f'Average accuracy on all corruptions: {(100*sum(test_acc_all)/len(test_acc_all))}')
    print(f'Min accuracy among all corruptions: {(100*min(test_acc_all))}')
    print(f'Max accuracy among all corruptions: {(100*max(test_acc_all))}')

    print(f'Top-5 accuracy on all corruptions: {([val*100 for val in test_acc_top5_all])}')
    print(f'Average top-5 accuracy on all corruptions: {(100*sum(test_acc_top5_all)/len(test_acc_top5_all))}')
    print(f'Min top-5 accuracy on all corruptions: {(100*min(test_acc_top5_all))}')
    print(f'Max top-5 accuracy on all corruptions: {(100*max(test_acc_top5_all))}')
       
    if all_severities:
        print(f'The above accuracies are averaged over all severities of each corruption. Below accuracies are for the highest severity of each corruption.')

    print(f'Severity level 5: Accuracy on all corruptions: {([val*100 for val in test_acc_sev5_all])}')
    print(f'Severity level 5: Average accuracy on all corruptions: {(100*sum(test_acc_sev5_all)/len(test_acc_sev5_all))}')
    print(f'Severity level 5: Min accuracy among all corruptions: {(100*min(test_acc_sev5_all))}')
    print(f'Severity level 5: Max accuracy among all corruptions: {(100*max(test_acc_sev5_all))}')

    print(f'Severity level 5: Top-5 accuracy on all corruptions: {([val*100 for val in test_acc_sev5_top5_all])}')
    print(f'Severity level 5: Average top-5 accuracy on all corruptions: {(100*sum(test_acc_sev5_top5_all)/len(test_acc_sev5_top5_all))}')
    print(f'Severity level 5: Min top-5 accuracy on all corruptions: {(100*min(test_acc_sev5_top5_all))}')
    print(f'Severity level 5: Max top-5 accuracy on all corruptions: {(100*max(test_acc_sev5_top5_all))}')


import tqdm

def standard_test(model, test_loader, verbose=True, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    test_correct_top5 = 0
    #max_to_mean_ratio = np.empty(0)

    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)

        
        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

        
        # Top-5 accuracy:
        test_correct_top5 += target.unsqueeze(dim=1).eq(torch.topk(output, k=5, dim=1)[1]).sum().item()

        # The following computes histogram of activation sparsity
        #hist_of_act_sparsity(model, data)


    test_size = len(test_loader.dataset)
    if verbose:
        print(
            f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}, Test acc top-5: {100*test_correct_top5/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size, test_correct_top5/test_size


def get_noisy_images(data, noise_std, noise_type="gaussian"):
       
    if noise_type=="gaussian":
        noise = torch.normal(mean=torch.zeros_like(
            data), std=noise_std*torch.ones_like(data))
    elif noise_type=="uniform":
        lims = noise_std * (3.0)**0.5
        noise = torch.zeros_like(data).uniform_(-lims, lims)
    else:
        raise NotImplementedError

    noisy_data = (data+noise).clamp(0.0, 1.0)

    return noisy_data





def test_noisy(model, test_loader, noise_std, noise_type="gaussian"):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    test_correct_top5 = 0

    snrs = torch.zeros(len(model.layer_inputs)).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                data = data[0]
                target = target[0]

            data, target = data.to(device), target.to(device)
            output = model(data)

            clean_layer_outs = list(model.layer_inputs.values())

            # noise = torch.normal(mean=torch.zeros_like(
            #     data), std=noise_std*torch.ones_like(data))

            # noisy_data = (data+noise).clamp(0.0, 1.0)
            noisy_data = get_noisy_images(data, noise_std, noise_type)

            output = model(noisy_data)

            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

            test_correct_top5 += target.unsqueeze(dim=1).eq(torch.topk(output, k=5, dim=1)[1]).sum().item()

            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)

            # breakpoint()
            noisy_layer_outs = list(model.layer_inputs.values())
            for layer_no, _ in enumerate(noisy_layer_outs):
                snrs[layer_no] += (torch.norm(clean_layer_outs[layer_no], p=2, dim=(1, 2, 3))/(torch.norm(noisy_layer_outs[layer_no]-clean_layer_outs[layer_no], p=2,dim=(1, 2, 3))+0.000000001)).square().sum()

    test_size = len(test_loader.dataset)
    test_acc = test_correct / test_size
    test_acc_top5 = test_correct_top5 / test_size

    snrs /= test_size
    snrs = snrs.tolist()
    snrs = [10*np.log10(s) for s in snrs]


    return test_acc, test_acc_top5, test_loss/test_size, snrs
