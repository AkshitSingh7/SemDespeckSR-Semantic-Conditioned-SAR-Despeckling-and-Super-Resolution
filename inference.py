"""Inference script for SAR-WaveSRNet."""
import torch
import numpy as np
import argparse
from models.generator import SARWaveSRNet


def inference(input_path, checkpoint_path, embed_path=None, scale=2, device='cuda'):
    """Run inference on a SAR amplitude patch.
    
    Args:
        input_path: Path to .npy amplitude patch
        checkpoint_path: Path to model checkpoint
        embed_path: Path to scene embedding .npy (512-dim)
        scale: Super-resolution scale
        device: torch device
    """
    model = SARWaveSRNet(scale=scale).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    patch = np.load(input_path).astype(np.float32)
    if patch.ndim == 2:
        patch = patch[np.newaxis, np.newaxis, ...]
    x = torch.from_numpy(patch).to(device)

    if embed_path is not None:
        embed = np.load(embed_path).astype(np.float32)
        embed = torch.from_numpy(embed).unsqueeze(0).to(device)
    else:
        embed = torch.zeros(1, 512).to(device)

    with torch.no_grad():
        output = model(x, embed).clamp(0, 1)

    return output.cpu().numpy().squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--embedding', default=None)
    parser.add_argument('--output', default='output.npy')
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    result = inference(args.input, args.checkpoint, args.embedding, args.scale)
    np.save(args.output, result)
    print(f"Saved output to {args.output} | Shape: {result.shape}")
