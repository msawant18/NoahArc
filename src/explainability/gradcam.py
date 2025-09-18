"""
Grad-CAM visualization.
"""
# src/explainability/gradcam.py
import numpy as np
import torch
import torch.nn.functional as F

def simple_cam(model, input_tensor, target_channel=0):
    """
    Very simple CAM: average absolute gradients of output wrt feature maps of bottleneck.
    This is a coarse explainability for UNet baseline.
    """
    model.eval()
    # register hook on bottleneck conv (we assume attribute name 'bottleneck')
    fmap = None
    grads = None
    def fmap_hook(m, inp, out):
        nonlocal fmap
        fmap = out.detach()
    def grad_hook(m, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()
    # attach
    if hasattr(model, "bottleneck"):
        h1 = model.bottleneck.register_forward_hook(fmap_hook)
        h2 = model.bottleneck.register_full_backward_hook(grad_hook)
    else:
        return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
    input_tensor = input_tensor.requires_grad_(True)
    out = model(input_tensor)
    # assume binary -> take mean logit
    score = out.mean()
    score.backward()
    if fmap is None or grads is None:
        return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
    weights = grads.mean(dim=(2,3), keepdim=True)  # global avg pool
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(input_tensor.shape[-2], input_tensor.shape[-1]), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # cleanup hooks
    h1.remove(); h2.remove()
    return cam
