import torch, random
from torch import nn
import numpy as np

def set_global_seed(seed):
    """
    Set global seed.

    :param seed: Random seed used for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def round_half_up(x):
    """
    Half-up rounding instead of built-in torch.round.

    :param x: Noise scaling factor; larger x gives relatively smaller Gaussian noise (sigma = mean_abs / x).
    :return: Tensor rounded in a half-up fashion with the same shape as x.
    """
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

def quantize_tensor_symmetric_round(w, q, A=None):
    """
    Symmetric, uniform quantization.

    For q >= 2 (even): total levels = q+1 including 0.

    :param w: Input weight tensor to be quantized or normalized.
    :param q: Number of quantization levels (power-of-two) for symmetric weight quantization.
    :param A: Symmetric quantization bound; weights are mapped into the range [-A, A].
    :return: Quantized tensor with the same shape as w.
    """
    if A is None:
        A = w.abs().max()
    if A == 0:
        return w

    # --- Original symmetric quantization for q >= 2 (even) ---
    n_side = (q - 2) // 2
    step = A / (n_side + 1)

    # Quantize to integer multiples of step
    q_int = round_half_up(w / step)

    w.copy_(q_int * step)
    return w

@torch.no_grad()
def quantize_all_weights(model, q, unfrozen=None):
    """
    Quantize all weights.

    :param model: PyTorch model whose parameters will be inspected or modified.
    :param q: Number of quantization levels (power-of-two) for symmetric weight quantization.
    :param unfrozen: Optional list of layer name substrings that are allowed to be modified (others treated as frozen).
    :return: The same model instance, after in-place quantization of selected layers.
    """
    allow_layers = None if unfrozen is None else set(unfrozen)
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if name == "snn.conv1.psp_func.weight":
            A = 0.3330
        elif name == "snn.conv2.psp_func.weight":
            A = 0.0417
        elif name == "snn.conv3.psp_func.weight":
            A = 0.0295
        elif name == "snn.temp_conv1.psp_func_list.0.weight":
            A = 0.0625
        elif name == "snn.temp_conv1.psp_func_list.1.weight":
            A = 0.0625
        elif name == "snn.temp_conv1.psp_func_list.2.weight":
            A = 0.0625
        elif name == "snn.rec1.rec_func.weight":
            A = 0.0625
        elif name == "snn.fc1.psp_func.weight":
            A = 0.0625
        elif name == "snn.fc2.weight":
            A = 0.0625
        else:
            A = None
            print(f"Warning: No A value found for layer {name}, using max abs value.")

        parts = name.split(".")
        if len(parts) < 3:
            continue
        _, layer_name, *rest = parts

        if (allow_layers is not None) and (layer_name not in allow_layers):
            continue  # frozen -> skip
        # print (f"Quantizing layer {layer_name} weights")
        quantize_tensor_symmetric_round(p.data, q, A)
    return model

@torch.no_grad
def add_gaussian_noise_levelwise(model, x=1.0, q=2, seed=None, unfrozen=None):
    """
    Add gaussian noise levelwise.

    :param model: PyTorch model whose parameters will be inspected or modified.
    :param x: Noise scaling factor; larger x gives relatively smaller Gaussian noise (sigma = mean_abs / x).
    :param q: Number of quantization levels (power-of-two) for symmetric weight quantization.
    :param seed: Random seed used for reproducibility.
    :param unfrozen: Optional list of layer name substrings that are allowed to be modified (others treated as frozen).
    :return: Dictionary mapping parameter names to dictionaries with keys 'mu', 'sigma' and 'post' describing the applied Gaussian noise.
    """
    if seed is not None:
        set_global_seed(seed)

    allow_layers = None if unfrozen is None else set(unfrozen)
    cache = {}

    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if name == "snn.conv1.psp_func.weight":
            A = 0.3330
        elif name == "snn.conv2.psp_func.weight":
            A = 0.0417
        elif name == "snn.conv3.psp_func.weight":
            A = 0.0295
        elif name == "snn.temp_conv1.psp_func_list.0.weight":
            A = 0.0625
        elif name == "snn.temp_conv1.psp_func_list.1.weight":
            A = 0.0625
        elif name == "snn.temp_conv1.psp_func_list.2.weight":
            A = 0.0625
        elif name == "snn.rec1.rec_func.weight":
            A = 0.0625
        elif name == "snn.fc1.psp_func.weight":
            A = 0.0625
        elif name == "snn.fc2.weight":
            A = 0.0625
        else:
            A = None
            print(f"Warning: No A value found for layer {name}, using max abs value.")

        parts = name.split(".")
        if len(parts) < 3:
            continue
        _, layer_name, *rest = parts

        if (allow_layers is not None) and (layer_name not in allow_layers):
            continue  # frozen -> skip

        w = p.data
        nz = p.detach().ne(0)
        # mu = w.abs().max()
        mu = w.abs()[nz].mean()
        sigma = mu / float(x)
        # print(f"Adding noise to layer {layer_name} weights with mu={mu:.6f}, sigma={sigma:.6f}")
        p.add_(torch.randn_like(w) * sigma)

        if A is not None:
            p.clamp_(-A, A)
            
        cache[name] = {"mu": mu, "sigma": sigma, "post": p.detach().clone()}
    return cache