
import numpy as np
import torch
import time

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution).cuda()
    x2 = torch.FloatTensor(1, 1, 256, 256).cuda()
    return dict(RGB=x1, depth=x2)


if __name__ == '__main__':
    # 加载网络
    from models.Network_IPAM import Architecture
    #加载模型
    RGBD_model = Architecture().cuda()
    #加载预训练权重
    RGBD_model.load_pretrained_model(RGBD_model.RGB_net, './pretrained/resnet_50.pth')
    RGBD_model.load_pretrained_model(RGBD_model.Depth_net, './pretrained/resnet_50.pth')
    #测试模型
    RGBD_model = RGBD_model.eval()

    #查看网络结构
    # from torchsummaryX import summary
    # summary(RGBD_model, torch.zeros(1, 3, 256, 256).cuda(), torch.zeros(1, 1, 256, 256).cuda())

    #计算网络的计算量
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(RGBD_model.cuda(), (3,256,256),
    #                                          input_constructor=prepare_input,
    #                                          as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #计算速度
    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 256, 256).cuda()

    time_spent = []
    for idx in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _, _, _, _ = RGBD_model(img, depth)

        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        end = time.time()

        if idx > 10:
            time_spent.append(end - start_time)

    print('Avg execution time (ms): {:.2f}'.format(np.mean(time_spent)*1000))
    print('FPS=',1/np.mean(time_spent))
    print(time_spent)



