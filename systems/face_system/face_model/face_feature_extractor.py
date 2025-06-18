import os
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LowRankLinear(nn.Module):
    """
    Implements a low-rank approximation of a linear layer using two smaller linear layers.
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


def apply_lora_to_model(model: nn.Module, rank_ratio: float = 0.2) -> None:
    """
    Recursively replaces Linear layers (except classification heads) with low-rank approximations.

    Args:
        model: The model to be modified.
        rank_ratio: Ratio of rank to original dimension (e.g., 0.2 = 20% rank).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "head" not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = module.bias is not None
            setattr(model, name, LowRankLinear(in_features, out_features, rank, bias))
        else:
            apply_lora_to_model(module, rank_ratio)


MODEL_CONFIG = {
    "edgeface_s_gamma_05": {"model_name": "edgenext_small", "rank_ratio": 0.5},
    "edgeface_xs_gamma_06": {"model_name": "edgenext_x_small", "rank_ratio": 0.6},
    "edgeface_base": {"model_name": "edgenext_base", "rank_ratio": None},
    "edgeface_xxs": {"model_name": "edgenext_xx_small", "rank_ratio": None},
}


class FaceFeatureExtractor(nn.Module):
    """
    Extracts L2-normalized feature embeddings from aligned face images using a lightweight vision transformer.
    """

    def __init__(
        self, extractor_arch: str, extractor_weights: str, device: str = "cuda"
    ):
        """
        Args:
            extractor_arch: Architecture key defined in MODEL_CONFIG.
            extractor_weights: Path to model weights (.pth).
            device: Device to load the model on.
        """
        super().__init__()

        if extractor_arch not in MODEL_CONFIG:
            raise ValueError(
                f"Unsupported feature architecture '{extractor_arch}'. Available: {list(MODEL_CONFIG.keys())}"
            )

        config = MODEL_CONFIG[extractor_arch]
        model = timm.create_model(config["model_name"])
        model.reset_classifier(512)  # Replace head with 512-dim output

        if config["rank_ratio"] is not None:
            apply_lora_to_model(model, rank_ratio=config["rank_ratio"])

        if not os.path.isfile(extractor_weights):
            raise FileNotFoundError(f"Weights not found: {extractor_weights}")

        state_dict = torch.load(extractor_weights, map_location=device)
        if any(k.startswith("model.") for k in state_dict.keys()):
            # Strip 'model.' prefix if present (e.g., from wrappers)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)

        self.backbone = model.to(device).eval()
        self.device = torch.device(device)

    @staticmethod
    def preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face image to model-ready tensor.

        Args:
            aligned_face: BGR uint8 image of shape (112, 112, 3)

        Returns:
            Float32 normalized CHW array.
        """
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return img

    @torch.no_grad()
    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract a L2-normalized embedding from a single aligned face image.

        Args:
            aligned_face: BGR image of size 112x112.

        Returns:
            1D numpy array (e.g., 512-dim feature vector).
        """
        img = self.preprocess(aligned_face)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        features = self.backbone(img_tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of preprocessed face tensors.

        Args:
            x: Tensor of shape (B, 3, 112, 112), already normalized.

        Returns:
            Tensor of shape (B, 512), L2-normalized.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if x.ndim != 4 or x.shape[1:] != (3, 112, 112):
            raise ValueError(f"Expected input shape (B, 3, 112, 112), got {x.shape}")

        if x.device.type != self.device.type:
            raise ValueError(f"Input tensor must be on device {self.device}")

        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)
