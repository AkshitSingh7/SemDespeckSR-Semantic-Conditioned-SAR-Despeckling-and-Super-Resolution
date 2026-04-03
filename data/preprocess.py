"""NISAR GCOV data preprocessing — extract patches from HDF5."""
import h5py
import numpy as np


def extract_patches_from_nisar(filepath, dataset_path='science/LSAR/GCOV/grids/frequencyA/HHHH',
                                row_slice=slice(6000, 14000), col_slice=slice(2000, 14000),
                                patch_size=128, stride=64, min_mean=0.05, min_std=0.02):
    """Extract normalized amplitude patches from NISAR GCOV HDF5 file.
    
    Args:
        filepath: Path to NISAR GCOV .h5 file
        dataset_path: HDF5 dataset path for backscatter data
        row_slice, col_slice: Region of interest
        patch_size: Patch dimensions
        stride: Stride between patches (< patch_size for overlap)
        min_mean, min_std: Thresholds to skip empty/invalid patches
    
    Returns:
        numpy array of patches (N, patch_size, patch_size), float32, [0,1]
    """
    with h5py.File(filepath, 'r') as f:
        data = f[dataset_path][row_slice, col_slice]

    # Convert power to amplitude
    amp = np.sqrt(np.clip(data, 0, None))

    # Percentile normalization
    p1, p99 = np.percentile(amp[amp > 0], [1, 99])
    amp_norm = np.clip((amp - p1) / (p99 - p1), 0, 1).astype(np.float32)

    patches = []
    for r in range(0, amp_norm.shape[0] - patch_size, stride):
        for c in range(0, amp_norm.shape[1] - patch_size, stride):
            patch = amp_norm[r:r + patch_size, c:c + patch_size]
            if patch.mean() > min_mean and patch.std() > min_std:
                patches.append(patch)

    return np.array(patches)
