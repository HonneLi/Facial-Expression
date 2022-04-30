import sys
import torch
import torchvision.models as models
from videocnn.models import resnet
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])


def get_model(type):
    assert type in ['2d', '3d']
    if type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    else:
        print('Loading 3D-Resnet-50 ...')
        model = resnet.resnet50(
            num_classes=3,
            shortcut_type='B',
            sample_size=224,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = torch.load(args.resnet50_model_path)
        model.load_state_dict(model_data)

    model.eval()
    print('Feature Extactor model loaded.')
    return model
