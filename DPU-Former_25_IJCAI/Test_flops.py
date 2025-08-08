import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network_UF_v2
#from lib.small_test1 import FFT2D
import torch
import torchvision
from thop import profile



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Model
print('==> Building model..')
input_features = torch.randn(1, 3, 352, 352)
model = Network_UF_v2(128)
#dummy_input = torch.randn(1, 2048, 12, 12)
flops, params = profile(model, (input_features,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


