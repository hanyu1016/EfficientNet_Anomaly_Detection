import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from torch.nn import Parameter
# from torchsummary import summary
from PIL import Image
# from abc import ABC, abstractmethod
from torchvision import datasets, transforms
from models.efficientNet import MyEfficientNet
import math
import os
import sys
sys.path.append('../')
import config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

class EmotionDetector():
    def __init__(self, modelIdx=18, fromPath=None):
        """
        Participants should define their own initialization process.
        During this process you can set up your network. The time cost for this step will
        not be counted in runtime evaluation
        """
        self.net = MyEfficientNet()
        if fromPath == None:
            checkpoint = torch.load(
                '%s/net_%03d.pth'%("./model/test_C+R_haveword_data_baseline",modelIdx),map_location='cuda:0')
        else:
            checkpoint = torch.load(fromPath,map_location='cuda:0')

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        state_dict = self.net.state_dict()
        
        for net_key, ckpt_key in zip(state_dict,checkpoint):
            if 'module.' in net_key and not 'module.' in ckpt_key:
                changeCkptName = True
                changeNetName = False
                break
            elif 'module.'not in net_key and 'module.' in ckpt_key:
                changeCkptName = False
                changeNetName = True
                break
            else:
                changeCkptName = False
                changeNetName = False
                break

        if changeCkptName:
            for weightName in checkpoint:
                netName = 'module.'+ weightName
                state_dict[netName]=checkpoint[weightName]
            self.net.load_state_dict(state_dict, strict=True)
        elif changeNetName:
            for weightName in checkpoint:
                netName = weightName[7:]
                state_dict[netName]=checkpoint[weightName]
            self.net.load_state_dict(state_dict, strict=True)
        else:
            self.net.load_state_dict(checkpoint, strict=True)
        # self.net.load_state_dict(checkpoint, strict=True)

        self.new_width = self.new_height = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.new_width, self.new_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        self.net.cuda()
        self.net.eval()
    def preprocess_data(self, image):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        return processed_data
    
    def eval_image(self, image):
        data = torch.stack(image, dim=0)
        channel = 3
        input_var = data.view(-1, channel, data.size(2), data.size(3)).cuda()
        with torch.no_grad():
            rst = self.net(input_var).detach()
        return rst

    def forwardWoM(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------        
        cosine = F.linear(F.normalize(input), F.normalize(self.head.weight))

        # --------------------------- convert label to one-hot ---------------------------
        # torch.cuda.set_device(1)
        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, label.view(-1, 1).long().cuda(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine) + ((1.0 - one_hot) * cosine)
        output *= 64

        return output
    
    def predict(self, images):
        """
        Process a list of image, the evaluation toolkit will measure the runtime of every call to this method.
        The time cost will include any thing that's between the image input to the final prediction score.
        The image will be given as a numpy array in the shape of (H, W, C) with dtype np.uint8.
        The color mode of the image will be **RGB**.
        
        params:
            - image (np.array): numpy array of required image
        return:
            - probablity (float)
        """
        real_data = []
        for image in images:
            data = self.preprocess_data(image)
            real_data.append(data)
        rst = self.eval_image(real_data)
        # theta = self.forwardWoM(rst,torch.zeros(len(rst)))
        probablity = torch.nn.functional.softmax(rst, dim=1).cpu().detach().numpy().copy()
        # probablity = F.softmax(theta, dim=1)
        probablity = np.array(probablity)
        return probablity