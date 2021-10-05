import torch
import torchvision

# 读入数据集
import dataset
trainloader, testloader, classes = dataset.dataset()

# 展示数据
import show_data
show_data.showdata(trainloader, classes)

# 加载网络结构
import lenet
net = lenet.Net()
print(net)

# 展示网络结构
from torchsummary import summary
summary(net, input_size=(3, 32, 32), device='cpu')

'''
params = list(net.parameters())
k=0
for i in params:
    l =1
    print("该层的结构："+str(list(i.size())))
    for j in i.size():
        l *= j
    print("参数和："+str(l))
    k = k+l

print("总参数和："+ str(k))
'''
