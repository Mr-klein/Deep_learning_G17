# Author: keith
# loads the output matrix from "loadplottest.py"
# and plots the confusion matrix for the classifier
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import torchviz as tv
from PIL import Image

confusion = np.load('outfile1.npy')
plt.xticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I',
            'J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z'])
plt.yticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I',
            'J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z'])


plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(confusion,cmap='PuBu',vmin=0., vmax=100.)
plt.Normalize(vmin=0.,vmax=100.)
plt.colorbar(pad=0.01).ax.set_title('prediction %')



plt.show()
