#real images
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import PIL
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pdb
from PIL import Image
import torch.nn.utils.spectral_norm as spectral_norm

use_cuda = torch.cuda.is_available()
# use_cuda = 1
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

dataset = torchvision.datasets.CocoCaptions("./data/", annFile)