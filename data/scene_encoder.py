"""CLIP-based scene encoder for generating semantic embeddings from SAR patches."""
import torch
import numpy as np
from torchvision import transforms


def encode_patches_with_clip(patches, model, batch_size=256, device='cuda'):
    """Encode SAR patches into CLIP embedding space.
    
    Args:
        patches: numpy array (N, H, W), float32, [0,1]
        model: CLIP model
        batch_size: Batch size for encoding
        device: torch device
    
    Returns:
        numpy array of embeddings (N, embed_dim)
    """
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batch_3ch = np.stack([batch, batch, batch], axis=1)
            batch_tensor = torch.from_numpy(batch_3ch).float()
            batch_tensor = clip_transform(batch_tensor).to(device)

            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def build_scene_prototypes(model, tokenizer, device='cuda'):
    """Build CLIP text embeddings for SAR scene categories.
    
    Uses expert-informed scene descriptions derived from SARLANG-1M
    caption analysis to create semantic prototypes.
    """
    categories = {
        'urban_dense': 'dense urban area with buildings roads and infrastructure',
        'urban_industrial': 'industrial area with warehouses factories and parking lots',
        'urban_residential': 'residential area with houses and streets',
        'water_body': 'water body river lake or ocean surface',
        'port_harbor': 'harbor or port with ships and docked boats',
        'vegetation': 'forest or dense vegetation and trees',
        'agricultural': 'agricultural fields farmland and crops',
        'desert_barren': 'desert or barren land with sparse vegetation',
        'coastal': 'coastal area where land meets water',
        'mixed_terrain': 'mixed terrain with roads buildings and natural features',
    }

    prototypes = {}
    with torch.no_grad():
        for name, desc in categories.items():
            tokens = tokenizer([desc]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            prototypes[name] = emb.cpu().numpy()

    return prototypes
