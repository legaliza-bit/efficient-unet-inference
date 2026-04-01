import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        )

    def forward(self, x):
        return self.model(x)["out"]


def get_pretrained_segmentation_model(num_classes: int = 21) -> nn.Module:
    return DeepLabV3Wrapper(num_classes=num_classes)
