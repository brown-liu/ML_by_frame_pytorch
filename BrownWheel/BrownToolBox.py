import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
from collections import OrderedDict
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class toolbox:
    @staticmethod
    def readimage(path, hasLabel,imreadflag,image_size):

        if imreadflag == 0:
            X = np.zeros((len(os.listdir(path)),1, image_size, image_size))
        else:
            X = np.zeros((len(os.listdir(path)), image_size, image_size, 3))
        y = np.zeros(len(os.listdir(path)))
        for idx, name in tqdm(enumerate(os.listdir(path))):
            file_path=os.path.join(path,name)
            image = cv2.imread(file_path,imreadflag)
            X[idx,0,:,:]=cv2.resize(image,(image_size,image_size))
            if hasLabel:
                y[idx]=int(name.split('_')[0])
        if hasLabel:
            return torch.from_numpy(X),torch.from_numpy(y)

        else:
            return torch.from_numpy(X)

    @staticmethod
    def count_correct(pred,label):
        return pred.argmax(dim=1).eq(label).sum().item()







