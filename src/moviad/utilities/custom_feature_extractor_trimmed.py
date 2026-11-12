"""
This class represent a feature extractor built on top of a custom backbone. The backbone name is passed as input to the constructor.
Based on the model name and the layers indexes it will consider the correct layers for the feature extraction.
"""

from __future__ import annotations
import logging
from pathlib import Path
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from moviad.backbones.mcunet.mcunet.model_zoo import net_id_list, build_model
from moviad.backbones.micronet.micronet import micronetBB
from moviad.utilities.get_sizes import count_params, params_to_mb

try:
    from micromind.networks.phinet import PhiNet
except ModuleNotFoundError:
    logging.error("micromind not found in current environment")

OTHERS_BACKBONES = (
    "mcunet-in3",
    "micronet-m0",
    "micronet-m1",
    "micronet-m2",
    "micronet-m3",
    "phinet_2.3_0.75_5",
    "phinet_1.2_0.5_6_downsampling",
    "phinet_0.8_0.75_8_downsampling",
    "phinet_1.3_0.5_7_downsampling",
    "phinet_0.9_0.5_4_downsampling_deep",
    "phinet_0.9_0.5_4_downsampling",
)

TORCH_BACKBONES = (
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
     "efficientnet_b4",
    "mobilenet_v2",
)


class CustomFeatureExtractor:

    def __init__(
        self,
        model_name: str,
        layers_idx: list,
        device: torch.device,
        frozen=True,
        quantized=False,
        calibration_dataloader=None,
    ):
        """
        Constructor

        Args:
            model_name (str): name of the backbone to use
            layers_idx (list): list of layers identifiers
            device (torch.device): device to be used
        """

        self.model_name = model_name
        self.quantized = quantized
        self.layers_idx = layers_idx
        self.device = device
        self.project_path = Path(__file__).parent.parent

        # ¢heck for backbone support
        if model_name not in OTHERS_BACKBONES + TORCH_BACKBONES:
            raise NotImplementedError(
                f"The backbone: {model_name} is not supported for feature extraction"
            )

        elif model_name in TORCH_BACKBONES:

            self.model = CustomFeatureExtractor.get_feature_extractor(
                model_name, layers_idx, frozen
            )

        else:
            # check for mcunet backbone
            if "mcunet" in model_name:
                if model_name in net_id_list:
                    self.model, _, _ = build_model(
                        net_id=self.model_name, pretrained=frozen
                    )

            # check for micronet backbone
            elif "micronet" in model_name:
                if frozen:
                    self.model = micronetBB(
                        self.device,
                        torch.load(
                            Path(
                                self.project_path,
                                f"backbones/micronet/weights/{self.model_name}.pth",
                            )
                        ),
                    )
                else:
                    self.model = micronetBB(self.device)

            # check for phinet backbone
            elif "phinet" in model_name:

                self.load_phinet(pretrained=frozen)

                if quantized:
                    self.load_quantized_phinet(calibration_dataloader)

            if not self.quantized:
                # trim the model till the last layer
                self.trim()

            self.attached = False

        self.model = self.model.to(self.device)

        if frozen:
            # freeze the model
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def load_phinet(self, pretrained=True):
        """
        This function handles the phinet backbone loading
        """

        if self.model_name == "phinet_2.3_0.75_5":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                alpha=2.3,
                beta=0.75,
                t_zero=5,
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            if pretrained:
                self.model.load_state_dict(
                    torch.load("backbones/phinet/new_phinet_small_71.pth.tar")[
                        "state_dict"
                    ]
                )
        elif self.model_name == "phinet_1.2_0.5_6_downsampling":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                num_layers=7,
                alpha=1.2,
                beta=0.5,
                t_zero=6,
                downsampling_layers=[4, 5, 7],
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            self.model.load_state_dict(
                torch.load(
                    Path(
                        self.project_path,
                        "backbones/phinet/new_phinet_divisor8_v2_downsampl.pt",
                    )
                )
            )
        elif self.model_name == "phinet_0.8_0.75_8_downsampling":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                num_layers=7,
                alpha=0.8,
                beta=0.75,
                t_zero=8,
                downsampling_layers=[4, 5, 7],
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            if pretrained:
                self.model.load_state_dict(
                    torch.load(
                        Path(
                            self.project_path,
                            "backbones/phinet/new_phinet_divisor8_v3.pth.tar",
                        )
                    )["state_dict"]
                )
        elif self.model_name == "phinet_1.3_0.5_7_downsampling":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                num_layers=7,
                alpha=1.3,
                beta=0.5,
                t_zero=7,
                downsampling_layers=[4, 5, 7],
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            if pretrained:
                self.model.load_state_dict(
                    torch.load(
                        Path(
                            self.project_path, "backbones/phinet/phinet_13057DS.pth.tar"
                        )
                    )["state_dict"]
                )
        elif self.model_name == "phinet_0.9_0.5_4_downsampling_deep":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                num_layers=9,
                alpha=0.9,
                beta=0.5,
                t_zero=4,
                downsampling_layers=[4, 5, 7],
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            if pretrained:
                self.model.load_state_dict(
                    torch.load(
                        Path(
                            self.project_path,
                            "backbones/phinet/phinet_09054DSDE.pth.tar",
                        )
                    )["state_dict"]
                )
        elif self.model_name == "phinet_0.9_0.5_4_downsampling":
            self.model = PhiNet(
                input_shape=(3, 224, 224),
                num_layers=7,
                alpha=0.9,
                beta=0.5,
                t_zero=4,
                downsampling_layers=[4, 5, 7],
                include_top=True,
                num_classes=1000,
                divisor=8,
            )
            if pretrained:
                self.model.load_state_dict(
                    torch.load("backbones/phinet/phinet_09054DS.pth.tar")["state_dict"]
                )

    def load_quantized_phinet(self, calibration_dataloader):
        from micromind.quantize import quantize_pt
        import os

        save_path = f"backbones/quantized/{self.model_name}/"
        os.makedirs(save_path, exist_ok=True)

        quantize_pt(
            self.model,
            save_path,
            calibration_loader=calibration_dataloader,
            test_loader=None,
            metrics=None,
        )

    @staticmethod
    def get_feature_extractor(backbone: str, return_nodes, pretrained=True):
        """Get the feature extractor from the backbone CNN.

        Args:
            backbone (str): Backbone CNN network
            return_nodes (list[str]): A list of return nodes for the given backbone.

        Raises:
            NotImplementedError: When the backbone is efficientnet_b5
            ValueError: When the backbone is not supported

        Returns:
            GraphModule: Feature extractor.
        """
        if pretrained:
            model = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1")
        else:
            model = getattr(torchvision.models, backbone)()

        return_nodes = {layer: layer for layer in return_nodes}

        feature_extractor = create_feature_extractor(
            model=model, return_nodes=return_nodes
        )

        return feature_extractor

    def trim(self):

        layers = None

        max_idx = int(self.layers_idx[-1]) + 1

        if "mcunet" in self.model_name:
            layers = [self.model.first_conv] + list(self.model.blocks)[:max_idx]

            # we must shift by one the layers ids because of the first conv layer
            self.layers_idx = [int(idx) + 1 for idx in self.layers_idx]
        elif "micronet" in self.model_name:
            layers = list(self.model.features)[:max_idx]
        elif "phinet" in self.model_name:
            layers = list(self.model._layers)[:max_idx]

        # actually trim the model
        self.model = torch.nn.Sequential(*layers)

    def attach(self):
        def hook(module, input, output):
            self.features.append(output)

        if self.quantized:
            layers = list(self.model.children())[0]
        else:
            layers = list(self.model.children())

        for idx in self.layers_idx:
            layers[int(idx)].register_forward_hook(hook)

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:

        if self.model_name in TORCH_BACKBONES:
            return list(self.model(batch).values())

        else:

            if not self.attached:
                self.attach()
                self.attached = True

            self.features = []
            self.model(batch.to(self.device))
            return self.features

    def get_size(self):
        return params_to_mb(count_params(self.model))
