#!/usr/bin/env python3

"""
anomaly_detector.py — AnomalyDetector razred za uporabo v ROS2 node.

Primer uporabe:
    detector = AnomalyDetector("output/best_model.pth", threshold=0.5)
    has_anomaly = detector.detect(cv2_image, save_path="rezultat.png")
"""

import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AnomalyDetector:
    def __init__(self, model_path: str, threshold: float = 0.5, device: str = None):
        """
        Parametri:
            model_path  : pot do best_model.pth
            threshold   : prag za binarizacijo maske (0.0–1.0)
            device      : 'cuda' / 'cpu' / None (samodejno)
        """
        self.threshold = threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.img_size = self._load_model(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        print(f"AnomalyDetector pripravljen | device={self.device} | "
              f"img_size={self.img_size} | threshold={self.threshold}")

    # ─────────────────────────────────────────────
    #  Javna metoda
    # ─────────────────────────────────────────────
    def detect(self, image: np.ndarray, save_path: str = None) -> bool:
        """
        Zazna anomalijo na sliki.

        Parametri:
            image     : BGR numpy array (kot ga vrne cv2.imread)
            save_path : če je podana, shrani poročilo (4-panel PNG) na to pot

        Vrne:
            True  — anomalija zaznana
            False — slika normalna
        """
        orig_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = (image.shape[1], image.shape[0])  # (width, height)

        prob_map = self._predict(orig_rgb)

        prob_resized = cv2.resize(prob_map, orig_size, interpolation=cv2.INTER_LINEAR)
        bin_mask     = (prob_resized > self.threshold).astype(np.uint8) * 255
        has_anomaly  = bool(bin_mask.any())

        if save_path:
            self._save_report(orig_rgb, prob_resized, bin_mask, has_anomaly, save_path)

        return has_anomaly

    # ─────────────────────────────────────────────
    #  Interno
    # ─────────────────────────────────────────────
    def _load_model(self, model_path: str):
        ckpt     = torch.load(model_path, map_location=self.device)
        args     = ckpt.get("args", {})
        backbone = args.get("backbone", "efficientnet-b4")
        img_size = args.get("img_size", 512)

        model = smp.Unet(
            encoder_name    = backbone,
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        ).to(self.device)

        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Model naložen: {backbone}, best_iou={ckpt.get('best_iou', 'N/A'):.4f}")
        return model, img_size

    @torch.no_grad()
    def _predict(self, rgb_array: np.ndarray) -> np.ndarray:
        """Vrne probability map (H, W) float32."""
        pil_img  = Image.fromarray(rgb_array)
        tensor   = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits   = self.model(tensor)
        prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
        return prob_map  # shape: (img_size, img_size)

    def _save_report(self, orig_rgb: np.ndarray, prob_map: np.ndarray,
                     bin_mask: np.ndarray, has_anomaly: bool, save_path: str):
        """Shrani 4-panel poročilo: original | heatmapa | maska | overlay."""
        overlay = orig_rgb.copy()
        anom_px = bin_mask > 127
        if anom_px.any():
            overlay[anom_px] = (
                overlay[anom_px] * 0.4 + np.array([255, 0, 0]) * 0.6
            ).astype(np.uint8)

        score = float(prob_map.max())
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"{'⚠ ANOMALIJA ZAZNANA' if has_anomaly else '✓ NORMALNO'} "
            f"| Maks. verjetnost: {score:.3f}",
            fontsize=14,
            color="red" if has_anomaly else "green",
            fontweight="bold",
        )

        axes[0].imshow(orig_rgb)
        axes[0].set_title("Originalna slika")
        axes[0].axis("off")

        im = axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Heatmapa verjetnosti")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(bin_mask, cmap="gray")
        axes[2].set_title(f"Binarna maska (prag={self.threshold})")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (rdeča = anomalija)")
        axes[3].axis("off")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
