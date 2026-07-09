"""
Simplified BEV perception model.

Takes 6 surround-view camera images and produces a BEV
(bird's eye view) prediction. Built with plain PyTorch
to understand the concept before adding design patterns.

Architecture:
    6 camera images → Backbone (ResNet18) → View Transform → BEV Decoder → BEV output
"""

import torch
import torch.nn as nn
import torchvision.models as models

from av_perception.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleBEVModel(nn.Module):
    """
    A minimal BEV perception model.

    Why nn.Module?
    Every PyTorch model inherits from nn.Module. It gives you:
    - Automatic parameter tracking (model.parameters())
    - GPU movement (model.to("cuda"))
    - Save/load (torch.save/torch.load)
    - Training/eval mode switching (model.train()/model.eval())

    Think of nn.Module as a contract: you define __init__ (what layers exist)
    and forward (how data flows through them). PyTorch handles everything else.
    """

    def __init__(
        self,
        num_cameras: int = 6,
        bev_height: int = 200,
        bev_width: int = 200,
        bev_channels: int = 64,
        output_channels: int = 3,
    ):
        """
        Args:
            num_cameras: Number of surround-view cameras (6).
            bev_height: Height of the BEV grid in pixels.
            bev_width: Width of the BEV grid in pixels.
            bev_channels: Number of feature channels in BEV space.
            output_channels: Number of output channels (3 for RGB BEV prediction).
        """
        super().__init__()

        self._num_cameras = num_cameras
        self._bev_h = bev_height
        self._bev_w = bev_width

        # ============================================================
        # STAGE 1: Backbone — extracts features from each camera image
        # ============================================================
        # We use a pretrained ResNet18 as our backbone.
        # ResNet18 takes a (3, 224, 224) image and produces (512, 7, 7) features.
        #
        # Why pretrained? The network has already learned to detect edges,
        # textures, shapes from millions of ImageNet images. We reuse that
        # knowledge instead of learning from scratch.
        #
        # Why ResNet18? It's small enough for your laptop GPU.
        # Larger models (ResNet50, EfficientNet) give better features
        # but need more memory.

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # We take everything except the final classification layers
        # (avgpool and fc), because we want spatial feature maps,
        # not a single class prediction.
        #
        # ResNet18 architecture:
        #   conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc
        #   (3,224,224) ──────────────────────────────────────→ (512,7,7) → (512) → (1000)
        #                                                         ↑
        #                                              we stop here (spatial features)

        self.backbone = nn.Sequential(
            resnet.conv1,      # (3, 224, 224) → (64, 112, 112)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # (64, 112, 112) → (64, 56, 56)
            resnet.layer1,     # (64, 56, 56) → (64, 56, 56)
            resnet.layer2,     # (64, 56, 56) → (128, 28, 28)
            resnet.layer3,     # (128, 28, 28) → (256, 14, 14)
            resnet.layer4,     # (256, 14, 14) → (512, 7, 7)
        )

        # Feature dimension coming out of ResNet18's layer4
        self._feat_dim = 512
        self._feat_h = 7
        self._feat_w = 7

        # ============================================================
        # STAGE 2: View Transform — projects image features to BEV grid
        # ============================================================
        # This is the KEY step that makes BEV perception work.
        #
        # The problem: we have features in IMAGE space (what the camera sees).
        # We need features in BEV space (top-down view of the world).
        #
        # In a full BEV model (like LSS or BEVDet), this involves:
        # - Predicting depth for each pixel
        # - Using camera intrinsics/extrinsics to project features to 3D
        # - Splatting those 3D features onto a BEV grid
        #
        # In our SIMPLIFIED version, we skip the geometric projection
        # and use a learned MLP to directly map flattened image features
        # to BEV space. This won't be geometrically accurate, but it
        # lets the network LEARN a mapping from camera views to BEV.
        #
        # Think of it as: instead of doing exact math to project camera
        # to BEV, we let the neural network figure out the relationship
        # from data. Less principled, but simpler to implement.

        # Each camera produces a (512, 7, 7) feature map = 512 * 7 * 7 = 25088 values
        # We compress this to a smaller representation first
        self.camera_compress = nn.Sequential(
            nn.Conv2d(self._feat_dim, bev_channels, kernel_size=1),  # (512, 7, 7) → (64, 7, 7)
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        # After compressing all 6 cameras: 6 * 64 * 7 * 7 = 18816 values
        # We map this to the BEV grid: 64 * 200 * 200 = 2,560,000 values
        # That's too big for a single linear layer (memory explosion).
        #
        # Instead, we use a small bottleneck:
        # flatten → compress → reshape to small BEV → upsample to full BEV

        small_bev_h = bev_height // 8  # 25
        small_bev_w = bev_width // 8   # 25

        self.view_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_cameras * bev_channels * self._feat_h * self._feat_w,
                      bev_channels * small_bev_h * small_bev_w),
            nn.ReLU(inplace=True),
        )

        self._small_bev_h = small_bev_h
        self._small_bev_w = small_bev_w
        self._bev_channels = bev_channels

        # ============================================================
        # STAGE 3: BEV Decoder — refines BEV features and predicts output
        # ============================================================
        # Takes the small BEV grid and upsamples it to full resolution
        # while refining features through convolution layers.
        #
        # ConvTranspose2d is "reverse convolution" — it upsamples
        # feature maps to larger spatial dimensions.
        # Think of it as the opposite of pooling: instead of shrinking,
        # it expands.

        self.bev_decoder = nn.Sequential(
            # (64, 25, 25) → (64, 50, 50)
            nn.ConvTranspose2d(bev_channels, bev_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),

            # (64, 50, 50) → (32, 100, 100)
            nn.ConvTranspose2d(bev_channels, bev_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bev_channels // 2),
            nn.ReLU(inplace=True),

            # (32, 100, 100) → (16, 200, 200)
            nn.ConvTranspose2d(bev_channels // 2, bev_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bev_channels // 4),
            nn.ReLU(inplace=True),

            # (16, 200, 200) → (3, 200, 200)
            nn.Conv2d(bev_channels // 4, output_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range to match normalized BEV target
        )

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SimpleBEVModel created: {total_params:,} total params, "
                    f"{trainable_params:,} trainable")

    def forward(self, cameras: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: cameras → BEV prediction.

        This is the method PyTorch calls when you do model(input).
        It defines HOW data flows through the network.

        Args:
            cameras: tensor of shape (B, 6, 3, H, W)
                     B = batch size
                     6 = number of cameras
                     3 = RGB channels
                     H, W = image height, width

        Returns:
            BEV prediction of shape (B, 3, bev_H, bev_W)
        """
        batch_size = cameras.shape[0]

        # --- Stage 1: Extract features from each camera ---
        # We process all cameras from all batches together for efficiency.
        # Reshape: (B, 6, 3, H, W) → (B*6, 3, H, W)
        B, N, C, H, W = cameras.shape
        imgs = cameras.view(B * N, C, H, W)

        # Run through backbone: (B*6, 3, 224, 224) → (B*6, 512, 7, 7)
        features = self.backbone(imgs)

        # Compress features: (B*6, 512, 7, 7) → (B*6, 64, 7, 7)
        features = self.camera_compress(features)

        # Reshape back to separate cameras: (B*6, 64, 7, 7) → (B, 6*64*7*7)
        features = features.view(batch_size, -1)

        # --- Stage 2: View Transform ---
        # (B, 6*64*7*7) → (B, 64*25*25)
        bev_features = self.view_transform(features)

        # Reshape to spatial BEV grid: (B, 64*25*25) → (B, 64, 25, 25)
        bev_features = bev_features.view(
            batch_size, self._bev_channels, self._small_bev_h, self._small_bev_w
        )

        # --- Stage 3: BEV Decoder ---
        # (B, 64, 25, 25) → (B, 3, 200, 200)
        bev_output = self.bev_decoder(bev_features)

        return bev_output
