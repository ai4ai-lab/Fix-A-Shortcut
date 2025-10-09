#!/usr/bin/env python
# model.py — CCM-style architecture with C (image), U (age/view), S (sex)

import torch
import torch.nn as nn
import torchvision.models as tv

# ---------------------------
# Encoders
# ---------------------------

class ConceptEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        inc = tv.inception_v3(
            weights=tv.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True   # <- must be True for this torchvision version
        )
        # drop AuxLogits and fc
        feats = []
        for name, m in inc.named_children():
            if name in ("AuxLogits", "fc"):
                continue
            feats.append(m)
        self.backbone = nn.Sequential(*feats)   # -> [B,2048,1,1]
        self.out_dim = 2048

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)            # [B,2048,1,1]
        return f.view(x.size(0), -1)    # [B,2048]



class UnknownEncoder(nn.Module):
    """
    z_U: 2-D -> 64-D MLP (age_norm, view_AP).
    """
    def __init__(self, in_dim=2, hid=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class ShortcutIdentity(nn.Module):
    """
    z_S: keep sex scalar as-is (recommended by CCM). Optionally pass through small MLP.
    """
    def __init__(self):
        super().__init__()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # expect shape [B] or [B,1]; return [B,1]
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return s.float()


# ---------------------------
# Head & full model
# ---------------------------

class LinearHead(nn.Module):
    """ Single linear layer over concatenated [C | U | S]. """
    def __init__(self, dim_c, dim_u, dim_s):
        super().__init__()
        self.fc = nn.Linear(dim_c + dim_u + dim_s, 1)

    def forward(self, z_c, z_u, z_s):
        z = torch.cat([z_c, z_u, z_s], dim=1)
        return self.fc(z).flatten()


class FullCCMModel(nn.Module):
    """
    End-to-end model:
      - concept_encoder -> 2048-D (C)
      - unknown_encoder -> 64-D  (U)
      - shortcut_identity -> 1-D (S)
      - linear head over concat
    """
    def __init__(self, freeze_c=True, use_unknown=True):
        super().__init__()
        self.concept_encoder = ConceptEncoder(pretrained=True, freeze=freeze_c)
        self.use_unknown = use_unknown
        self.unknown_encoder = UnknownEncoder(in_dim=2, hid=64, out_dim=64) if use_unknown else None
        self.shortcut_encoder = ShortcutIdentity()
        dim_u = 64 if use_unknown else 0
        self.head = LinearHead(self.concept_encoder.out_dim, dim_u, dim_s=1)

    def forward(self, image, sex_scalar, unknown_feats=None):
        """
        image: [B,3,224,224] (or 299 if you retool your tfms for Inception)
        sex_scalar: [B] or [B,1]  (0=F, 1=M)
        unknown_feats: [B,2] (age_norm, view_AP) if use_unknown=True
        """
        z_c = self.concept_encoder(image)
        z_s = self.shortcut_encoder(sex_scalar)
        if self.use_unknown:
            if unknown_feats is None:
                raise ValueError("unknown_feats is required when use_unknown=True")
            z_u = self.unknown_encoder(unknown_feats)
        else:
            z_u = torch.empty(image.size(0), 0, device=image.device)
        logits = self.head(z_c, z_u, z_s)
        return logits

    # convenient slice getters for EYE/CBM regularizers
    def head_weight_slices(self):
        """
        Return weight tensors for (w_C, w_U, w_S) from the linear head.
        Useful for EYE/CBM penalties.
        """
        w = self.head.fc.weight.squeeze(0)  # [dim_total]
        dim_c = self.concept_encoder.out_dim
        dim_u = self.unknown_encoder.out_dim if self.use_unknown else 0
        dim_s = 1
        w_c = w[:dim_c]
        w_u = w[dim_c:dim_c+dim_u] if dim_u > 0 else torch.tensor([], device=w.device)
        w_s = w[-dim_s:]
        return w_c, w_u, w_s
