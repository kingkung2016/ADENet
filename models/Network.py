
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Backbone import resnet
from models import Attention_module


class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()

        self.RGB_net = resnet.resnet50(include_top=False)
        self.Depth_net = resnet.resnet50(include_top=False)

        self.image_size = 256
        self.in_channels = [256, 512, 1024, 2048]
        self.out_channels = [64, 128, 256, 512]

        self.fusion_1 = Attention_module.Cross_Modality_Fusion(self.in_channels[0], self.out_channels[0])
        self.fusion_2 = Attention_module.Cross_Modality_Fusion(self.in_channels[1], self.out_channels[1])
        self.fusion_3 = Attention_module.Cross_Modality_Fusion(self.in_channels[2], self.out_channels[2])
        self.fusion_4 = Attention_module.Cross_Modality_Fusion(self.in_channels[3], self.out_channels[3])

        #解码器
        self.decoder = Attention_module.PMA_decoder(self.out_channels)


    def load_pretrained_model(self, model, pretrained_path):
        #加载权重参数
        pretrained_dict = torch.load(pretrained_path)
        #获取当前模型参数
        model_dict = model.state_dict()
        #将pretained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #更新模型权重
        model_dict.update(pretrained_dict)
        #新的权重加载到network中
        model.load_state_dict(model_dict)


    def forward(self, RGB, depth):
        #主干网络提取特征
        R1, R2, R3, R4 = self.RGB_net(RGB)
        D1, D2, D3, D4 = self.Depth_net(depth)
        #256*64*64, 512*32*32, 1024*16*16, 2048*8*8

        feature_1 = self.fusion_1(R1, D1)
        feature_2 = self.fusion_2(R2, D2)
        feature_3 = self.fusion_3(R3, D3)
        feature_4 = self.fusion_4(R4, D4)

        out_1, out_2, out_3, out_4 = self.decoder(feature_1, feature_2, feature_3, feature_4)

        return out_1, out_2, out_3, out_4


if __name__== '__main__':
    rgb = torch.rand(1,3,256,256).cuda()
    depth = torch.rand(1,3,256,256).cuda()

    RGBD_model = Architecture().cuda()
    RGBD_model.load_pretrained_model(RGBD_model.RGB_net, '../pretrained/resnet_50.pth')
    RGBD_model.load_pretrained_model(RGBD_model.Depth_net, '../pretrained/resnet_50.pth')

    F1, F2, F3, F4 = RGBD_model(rgb, depth)
    print(F1.shape, F2.shape, F3.shape, F4.shape)




