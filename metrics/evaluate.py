"""Evaluation metrics for SAR despeckling and super-resolution."""
import torch
import torch.nn.functional as F


def calc_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-8))


def calc_ssim(pred, target, window_size=11):
    """Structural Similarity Index (higher is better, max 1.0)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_t = F.avg_pool2d(target, window_size, 1, window_size // 2)
    sig_pp = F.avg_pool2d(pred * pred, window_size, 1, window_size // 2) - mu_p * mu_p
    sig_tt = F.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu_t * mu_t
    sig_pt = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_pp + sig_tt + C2))
    return ssim.mean()


def calc_enl(img, patch_size=32):
    """Equivalent Number of Looks — measures speckle suppression (higher is better)."""
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    mean = patches.mean(dim=(1, 2))
    var = patches.var(dim=(1, 2)) + 1e-8
    return (mean ** 2 / var).mean()


def calc_epd(pred, target):
    """Edge Preservation Degree — measures structural retention (closer to 1.0 is better)."""
    sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    edge_p = F.conv2d(pred, sobel, padding=1).abs()
    edge_t = F.conv2d(target, sobel, padding=1).abs()
    return edge_p.sum() / (edge_t.sum() + 1e-8)
