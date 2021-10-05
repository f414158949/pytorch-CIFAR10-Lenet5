import torch
import torchvision
import torchvision.transforms as transforms

def dataset():
    transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 如果在windows下运行或者出现BrokenPipeError，那么将torch.utils.data.DataLoader()的num_workers赋值为0.
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=0) # num_workers改为0

    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=0) # num_workers改为0
    print('train_dataset 样本数量', len(trainset))
    print('train_dataset batch的数量', len(trainloader))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes