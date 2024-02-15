from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn as nn
import copy
import os.path
import pandas as pd
import os
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

NumCell = 30  # number of cells
NumClass = 6  # number of classes
save_name = 'mobilenet_indoor'  # name of the model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer and learning rate
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)