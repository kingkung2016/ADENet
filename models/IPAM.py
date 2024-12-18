import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import colorsys


# -----Filter相关的基础函数------
def rgb2lum(image):
    image = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    return image[:, None, :, :]

def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5

def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):
        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


def lerp(a, b, l):
    return (1 - l) * a + l * b


# -----Filter的相关实现------
class Filter(nn.Module):

    def __init__(self, net, cfg):
        super(Filter, self).__init__()

        self.cfg = cfg
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
               features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param, defog, IcA):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    def apply(self,
              img,
              img_features=None,
              defog_A=None,
              IcA=None,
              specified_parameter=None,
              high_res=None,
              net=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:  ##########进了
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)    #######这块就是把这个参数，然后输入进滤波器中，让他们自己去调节，最后输出的想要的参数。
            # print(filter_features)
            # print(filter_parameters)
        else:
            raise NotImplementedError

        low_res_output = self.process(img, filter_parameters, defog_A, IcA)
        return low_res_output, filter_parameters


class ExposureFilter(Filter):         ######曝光滤波器

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'E'
        self.begin_filter_parameter = cfg.exposure_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):  # param is in (-self.cfg.exposure_range, self.cfg.exposure_range)
        return tanh_range(
            -self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

    def process(self, img, param, defog, IcA):
        return img * torch.exp(param * np.log(2))


class GammaFilter(Filter):  # gamma_param is in [1/gamma_range, gamma_range]       #gamma

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param, defog_A, IcA):
        #print('      param:', param)

        # param_1 = param.repeat(1, 3)
        zero = torch.zeros_like(img) + 0.00001
        img = torch.where(img <= 0, zero, img)
        # print("GAMMMA", param)
        return torch.pow(img, param)


class UsmFilter_FA(Filter):  # Usm_param is in [Defog_range]

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        def make_gaussian_2d_kernel(sigma):
            radius = 12
            x = torch.arange(-radius, radius + 1)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / torch.sum(k)
            return k.unsqueeze(1) * k

        kernel_i = make_gaussian_2d_kernel(5)

        kernel_i = kernel_i.clone().detach().to(img.device)
        # print('kernel_i.shape', kernel_i.shape)
        kernel_i = kernel_i.unsqueeze(0)
        kernel_i = kernel_i.unsqueeze(0)

        output = F.conv2d(img, weight=kernel_i, stride=1, groups=1, padding=12)
        img_out = (img - output) * param + img
        # img_out = (img - output) * 2.5 + img

        return img_out


class UsmFilter(Filter):  # Usm_param is in [Defog_range]      #锐化

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        self.channels = 1
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)

        # print('      param:', param)

        kernel = kernel.to(img.device)

        output = F.conv2d(img, kernel, padding=2, groups=self.channels)

        img_out = (img - output) * param + img
        # img_out = (img - output) * torch.tensor(0.043).cuda() + img

        return img_out


class ContrastFilter(Filter):        ####对比度

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param

        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.cont_range)(features)

    def process(self, img, param, defog, IcA):
        # print('      param.shape:', param.shape)

        # luminance = torch.minimum(torch.maximum(rgb2lum(img), 0.0), 1.0)
        #luminance = rgb2lum(img)
        luminance = img #单通道直接复制
        zero = torch.zeros_like(luminance)
        one = torch.ones_like(luminance)

        luminance = torch.where(luminance < 0, zero, luminance)
        luminance = torch.where(luminance > 1, one, luminance)

        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param)
        # return lerp(img, contrast_image, torch.tensor(0.015).cuda())


class ToneFilter(Filter):      #色调

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(features)
        return tone_curve

    def process(self, img, param, defog, IcA):
        param = torch.unsqueeze(param, 3)
        #print('      param.shape:', param.shape)

        tone_curve = param
        tone_curve_sum = torch.sum(tone_curve, axis=1) + 1e-30
        #print('      tone_curve_sum.shape:', tone_curve_sum.shape)

        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image = total_image + torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * param[:, i, :, :, :]

        total_image *= self.cfg.curve_steps / tone_curve_sum
        img = total_image
        return img


class LevelFilter(Filter):
    def __init__(self, net, cfg):
        super(LevelFilter, self).__init__(net, cfg)
        self.short_name = 'Le'
        self.begin_filter_parameter = cfg.level_begin_param
        self.num_filter_parameters = 2

    def filter_param_regressor(self, features):
        return torch.sigmoid(features)

    def process(self, img, param, defog, IcA):
        lower = param[:, 0]
        upper = param[:, 1] + 1
        lower = lower[:, None]
        upper = upper[:, None]

        result = torch.clamp((img - lower) / (upper - lower + 1e-6), 0.0, 1.0)

        # print(result)
        # print(result.shape)

        return result


# ----------Filter模块的参数------------
from easydict import EasyDict as edict

cfg = edict()
cfg.num_filter_parameters = 4
# 这里的配置均被用于DIF模块的滤波操作
# 定义了在DIF模块中各个滤波器参数在参数列表中的起始位置
cfg.exposure_begin_param = 0
cfg.gamma_begin_param = 1
cfg.contrast_begin_param = 2
cfg.tone_begin_param = 999
cfg.usm_begin_param = 3

cfg.exposure_range = 3.5  # 曝光参数的范围
# Gamma = 1/x ~ x
# Gamma参数范围为1/x到x，用于Gamma校正滤波器
cfg.gamma_range = 3  # Gamma校正参数的范围
cfg.usm_range = (0.0, 5)  # 锐化参数的范围
cfg.cont_range = (0.0, 1.0)  # 对比度参数的范围
cfg.wb_range = 1.1 #白平衡的范围
cfg.curve_steps = 8  # 色调曲线的步数
cfg.tone_curve_range = (0.5, 2)  # 色调曲线的范围
cfg.color_curve_range = (0.90, 1.10)  # 颜色曲线的范围

cfg.lab_curve_range = (0.90, 1.10)  # LAB颜色空间曲线的范围
cfg.defog_range = (0.1, 1.0)  # 去雾参数的范围



# ----------DIF模块------------
class DIF(nn.Module):
    def __init__(self, Filters):
        super(DIF, self).__init__()
        self.Filters = Filters

    def forward(self, img_input, Pr, img_depth):
        # 初始化滤波后的图像批次
        self.filtered_image_batch = img_depth
        # 对所有滤波器应用配置参数，生成滤波器列表
        filters = [x(img_input, cfg) for x in self.Filters]
        #print(filters)   #这个输出的是[ExposureFilter(), GammaFilter(), ContrastFilter(), UsmFilter()]
        # 初始化滤波器参数和滤波后的图像列表
        self.filter_parameters = []
        self.filtered_images = []
        # 遍历每个滤波器并应用
        for j, filter in enumerate(filters):
            #print(j)
            # 对滤波器应用到图像批次，获取滤波参数
            self.filtered_image_batch, filter_parameter = filter.apply(self.filtered_image_batch, Pr)

            # 将滤波参数和滤波后的图像添加到列表中
            self.filter_parameters.append(filter_parameter)
            #print(self.filtered_image_batch.type)
            self.filtered_images.append(self.filtered_image_batch)
            #print(len(self.filtered_image_batch))   每一组滤波器会修改四张图
            #print(len(self.filtered_images))

        # 返回滤波后的图像批次、滤波后的图像列表、配置参数和滤波器参数
        return self.filtered_image_batch, self.filtered_images, Pr, self.filter_parameters

    # ----------IPAM模块------------


def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class IPAM(nn.Module):
    def __init__(self):
        super(IPAM, self).__init__()

        # 定义预处理CNN模型
        self.CNN_PP = nn.Sequential(
            # 上采样至 (256, 256) 尺寸，双线性插值
            nn.Upsample(size=(256, 256), mode='bilinear'),
            # 卷积层，输入通道数为 3，输出通道数为 16，核大小为 3，步长为 2，padding 为 1
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            # LeakyReLU 激活函数，参数为 0.2
            nn.LeakyReLU(0.2),
            # 实例标准化，通道数为 16，可学习参数
            nn.InstanceNorm2d(16, affine=True),
            # 使用自定义函数 conv_downsample 创建一系列卷积层和下采样操作，通道数从 16 逐步增加至 128
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True),
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128),
            # Dropout 层，概率为 0.5
            nn.Dropout(p=0.5),
            # 最后一层卷积层，输入通道数为 128，输出通道数为配置参数中定义的滤波器参数数量
            nn.Conv2d(128, cfg.num_filter_parameters, 8, padding=0),
        )####最后一步经历了滤波器的参数，所以我要去看看滤波器的设置是什么，它说经过CNN_PP输出的应该是特征图
        #####论文中是这么说的：滤波器的超参数由小型基于CNN的参数预测器（CNN-PP）根据输入图像的亮度、对比度和曝光信息自适应预测。
        ###所以CNN_PP输出的是我滤波器的参数设置。

        # 定义一系列滤波器
        Filters = [ExposureFilter, GammaFilter, ContrastFilter, UsmFilter]

        # 初始化 DIF 模块，并传入滤波器列表
        self.dif = DIF(Filters)

    def forward(self, img_input, img_depth):
        # 使用预处理 CNN 模型处理输入图像，获取配置参数
        self.Pr = self.CNN_PP(img_input)    #通过CNN-PP得到的是rgb图像的特征图
        #print(self.Pr)
        ###这个输出的确实是tensor张量，并且跟输出的filter_parameter的大小保持一致。
        # 将输入图像和配置参数传入 DIF 模块进行处理
        out = self.dif(img_input, self.Pr, img_depth)
        # 返回处理结果
        return out


img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

depths_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

# 显示多个图像
def myimshows(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        #plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        img = img.squeeze(0).permute(1, 2, 0)
        plt.imshow(img.cpu().detach().numpy())
        plt.title(title)
        plt.axis('off')
    plt.show()

if __name__== '__main__':

    rgb = "../Visualization/6106511202_374c343594_b.jpg"
    depth = "../Visualization/6106511202_374c343594_b.png"

    # 读取图像并转换为张量
    ori_RGB = Image.open(rgb).convert('RGB')
    RGB_img = img_transform(ori_RGB).unsqueeze(0)
    n, c, h, w = RGB_img.size()  # batch_size, channels, height, weight

    ori_depth = Image.open(depth).convert('L')
    Depth_img = depths_transform(ori_depth).unsqueeze(0)
    Depth_copy = Depth_img.view(n, 1, h, w).repeat(1, c, 1, 1)  # 把深度图变成3个通道

    Preprocess = IPAM().cuda()
    RGB_tensor = RGB_img.cuda()
    Depth_tensor = Depth_img.cuda()
    Depth_copy = Depth_copy.cuda()

    filtered_image_batch_R, filtered_images_R, Pr, filter_parameters = Preprocess(RGB_tensor, Depth_tensor)
    filtered_image_batch_D, filtered_images_D, _, _ = Preprocess(Depth_copy, Depth_tensor)

    print(Depth_tensor)
    print(filtered_image_batch_R)
    print(filtered_image_batch_D)

    # 创建图像和标题列表
    images = [Depth_img, filtered_image_batch_R, filtered_image_batch_D]
    titles = ["Original Image", "Enhanced Image", "Enhanced Depth"]

    # 显示图像
    myimshows(images, titles)




