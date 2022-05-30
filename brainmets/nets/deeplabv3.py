import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class deeplabv3_50(nn.Module):
    def __init__(self, in_chan=3, out_chan=2, pretrained=True):
        super(deeplabv3_50, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, pretrained_backbone=pretrained)
        self.model.classifier = DeepLabHead(2048, out_chan)
        if in_chan != 3:
            self.model.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.model(x)['out']