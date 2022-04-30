import torch
import math
import numpy as np
from Preprocess import RandomDataset
from torch.utils.data import DataLoader
import argparse
from model import get_model
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='0. Video feature extractor')

parser.add_argument(
    '--video_path',
    type=str,
    help='input video path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='3d',
                            help='input type')
parser.add_argument('--half_precision', type=bool, default=True,
                            help='output half precision float')
parser.add_argument('--l2_normalize', type=bool, default=True,
                            help='l2 normalize feature')
parser.add_argument('--resnet_model_path', type=str, default='model/resnet50.pth',
                            help='Resnet model path')
args = parser.parse_args()


transform = [transforms.Resize((224, 224))]
frame_transform = transforms.Compose(transform)
dataset = RandomDataset(
  root=args.video_path,
  frame_transform=frame_transform
)
loader = DataLoader(
  dataset, 
  batch_size=args.batch_size,
  shuffule=True
)
model = get_model(args.type)

with torch.no_grad():
    for batch in loader:
        for i in range(len(batch['path'])):
            input_tensor = batch['video'][i]
            features = model(input_tensor)
            if args.l2_normalize:
                features = F.normalize(features, dim=1)
            features = features.cpu()
            if args.half_precision:
                features = features.astype('float16')
            batch['video'][i] = features
            print('Video {} features extracted.'.format(batch['path'][i]))
