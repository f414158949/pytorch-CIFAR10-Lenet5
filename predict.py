import torch
import torchvision
# 加载一个提前训练好的模型模型（10 个epoch）
'''
pytorch有两种模型保存方式：
一、保存整个神经网络,save的对象是网络net
二、只保存网络的参数,速度快,占空间少,save的对象是net.state_dict()
'''
from lenet import Net
#net = torch.load('./model/model10.pkl')
net = Net()
net.load_state_dict(torch.load('./model/mymodel.pkl'))

# 1显示测试集中的图片

import dataset
trainloader, testloader, classes = dataset.dataset() # 读入数据集

# 显示图片
dataiter = iter(testloader)
images, labels = dataiter.next()

import show_data
show_data.imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 让我们看看神经网络认为以上图片是什么。
outputs = net(images)
# 输出是10个标签的概率。 一个类别的概率越大，神经网络越认为它是这个类别。所以让我们得到最高概率的标签。
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# 2看看网络在整个测试集上的结果如何。

# 查看总体结果
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 查看各类的结果
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))