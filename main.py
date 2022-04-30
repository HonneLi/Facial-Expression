import torch
import torchvision
import math
import numpy as np
from Preprocess import RandomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from model import get_model
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='MS-TCN trained with extracted features')

parser.add_argument(
    '--root',
    type=str,
    help='input video path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='3d',
                            help='input type')
parser.add_argument('--action', type=str, default='train',
                            help='input type')
parser.add_argument('--half_precision', type=bool, default=False,
                            help='output half precision float')
parser.add_argument('--l2_normalize', type=bool, default=True,
                            help='l2 normalize feature')
parser.add_argument('--resnet_model_path', type=str, default='model/resnet50.pth',
                            help='Resnet model path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_transform = transforms.Compose([transforms.Resize((224, 224))])

dataset = RandomDataset(
  args,
  frame_transform=frame_transform
)

loader = DataLoader(
  dataset, 
  batch_size=args.batch_size,
  shuffule=True
)

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
