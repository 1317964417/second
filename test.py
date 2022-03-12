import torch
import torch.nn as nn
from model.resnet18 import Resnet18
from cifar10 import cifar10_data_download

data_loader_train, data_loader_test = cifar10_data_download(100)

device = torch.device("cuda")
net = Resnet18().to(device)
net.load_state_dict(torch.load("./modelsaved/resnet18_050_fgsm_epsilon=01.pth"))
net.eval()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


with torch.no_grad():
    correct = 0
    total = 0
    for data in data_loader_test:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))

