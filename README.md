# pytorch-CIFAR10-Lenet5

入坑github作业：在pytorch框架下用LeNet5在CIFAR10数据集上实现分类

模型选用Lenet 准确率在60%左右

### **作业要求**

Homework #1
Creating your own github account.
Implementing your own deep neural network (in Pytorch,PaddlePaddle...).
Training it on CIFAR10.
Tuning a hyper-parameter and analyzing its effects on performance.
Writing a README.md to report your findings.



### 文件夹说明

dataset  （存放数据集压缩包cifar-10-python.tar.gz,需要手动下载该数据集并放到该文件夹中）

model  （存放模型）

drawpic （存放readme里的图片）



### 调参分析

**1、learning rate**

<img src=".\drawpic\Figure_lr.png" alt="Figure_lr" style="zoom: 80%;" />

学习率过大：导致模型无法收敛

学习率过小：模型收敛速度较慢



**2、momentum**

<img src=".\drawpic\Figure_momentum.png" alt="Figure_momentum" style="zoom: 80%;" />

当![\公测](https://private.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cbeta)= 0.9时，最大速度相当于梯度下降的10倍（带进上式去算可得），通常![\公测](https://private.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cbeta)可取0.5,0.9,0.99，情况一般![\公测](https://private.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cbeta)的调整没有![\α](https://private.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Calpha)调整的那么重要适当取值即可。



**3、batchsize**

<img src=".\drawpic\Figure_batchsize.png" alt="Figure_batchsize" style="zoom: 80%;" />

batch_size常取2的n次方。 取 32 64 128 256 … 。batch越大，一般模型加速效果明显。尽管batch size越大，模型运行越快，但并不是越大，效果越好。

batch size增大，模型训练越稳定，学习率可以些许调大，但是有时候小的batch size带来的噪声影响可能会使模型获得更好的泛化能力，更容易越过损失函数中的鞍点（损失函数在某个点的某个方向呈现已到达最小值的假象，其实在另一个方向是万丈深渊，这是小batch提供的随机性能够帮助模型跳过鞍点。），当然小batch size存在训练稳定性差的缺点。
