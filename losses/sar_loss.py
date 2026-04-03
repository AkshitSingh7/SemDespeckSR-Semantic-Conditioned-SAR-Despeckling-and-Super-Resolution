"""SAR-specific loss functions for joint despeckling and super-resolution."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ENLLoss(nn.Module):
    """Equivalent Number of Looks loss — penalises speckle in homogeneous regions."""
    def __init__(self, patch_size=16):
        super().__init__()
        self.ps = patch_size

    def forward(self, pred):
        B, C, H, W = pred.shape
        patches = pred.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        patches = patches.contiguous().view(-1, self.ps, self.ps)
        mean = patches.mean(dim=(1, 2))
        var = patches.var(dim=(1, 2)) + 1e-8
        return -(mean ** 2 / var).mean()


class SSIMLoss(nn.Module):
    """Structural Similarity loss for preserving image structure."""
    def __init__(self, window_size=11):
        super().__init__()
        self.ws = window_size

    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = F.avg_pool2d(pred, self.ws, 1, self.ws // 2)
        mu_t = F.avg_pool2d(target, self.ws, 1, self.ws // 2)
        sig_pp = F.avg_pool2d(pred * pred, self.ws, 1, self.ws // 2) - mu_p * mu_p
        sig_tt = F.avg_pool2d(target * target, self.ws, 1, self.ws // 2) - mu_t * mu_t
        sig_pt = F.avg_pool2d(pred * target, self.ws, 1, self.ws // 2) - mu_p * mu_t
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_pp + sig_tt + C2))
        return 1 - ssim.mean()


class EdgeLoss(nn.Module):
    """Edge preservation loss using Sobel gradients."""
    def forward(self, pred, target):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        pred_edge = F.conv2d(pred, sobel_x, padding=1).abs() + F.conv2d(pred, sobel_y, padding=1).abs()
        tgt_edge = F.conv2d(target, sobel_x, padding=1).abs() + F.conv2d(target, sobel_y, padding=1).abs()
        return F.l1_loss(pred_edge, tgt_edge)


class SARWaveSRLoss(nn.Module):
    """Combined loss: L1 + SSIM + Edge + optional adversarial."""
    def __init__(self, l1_w=1.0, ssim_w=0.5, edge_w=0.3, adv_w=0.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.l1_w, self.ssim_w, self.edge_w, self.adv_w = l1_w, ssim_w, edge_w, adv_w

    def forward(self, pred, target, disc_fake=None):
        loss = self.l1_w * self.l1(pred, target) + \
               self.ssim_w * self.ssim(pred, target) + \
               self.edge_w * self.edge(pred, target)
        if disc_fake is not None and self.adv_w > 0:
            loss += self.adv_w * (-disc_fake.mean())
        return loss
