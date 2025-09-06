import math
from opacus import PrivacyEngine

def attach_privacy(model, optimizer, data_loader, noise_multiplier=1.1, max_grad_norm=1.0, target_delta=1e-5, sample_rate=None):
    pe = PrivacyEngine()
    model, optimizer, data_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=None,  # compute epsilon after training
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
    )
    return model, optimizer, data_loader, pe

def epsilon_after(pe, num_steps, sample_rate, target_delta):
    if sample_rate is None:
        return float('nan')
    eps = pe.accountant.get_epsilon(delta=target_delta)
    return eps
