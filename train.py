# 读入数据集
import dataset
trainloader, testloader, classes = dataset.dataset()

# 加载网络结构
import lenet
net = lenet.Net()
import torch
'''
import os
if os.path.exists("./model/mymodel.pkl"):
    net.load_state_dict(torch.load("./model/mymodel.pkl"))
'''

# 定义损失函数和优化器
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练网络
import time # 记录训练时间
start = time.time()



for epoch in range(5):  # 多批次循环

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        if i % 1000 == 999:    # 每1000批次打印一次， 在 gpu 上训练请调大此参数
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0


print('Finished Training')
print('程序的运行时间：%.2f' % (time.time() - start), "s")
# 快速保存我们训练好的模型：
torch.save(net.state_dict(), "./model/mymodel.pkl")
print('model saved')
