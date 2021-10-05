import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def showdata(trainloader, classes):
    # 获取随机数据
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # 显示图像标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # 展示图像
    imshow(torchvision.utils.make_grid(images))