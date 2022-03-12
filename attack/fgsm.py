import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.resnet18 import Resnet18
from cifar10 import cifar10_data_download
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent



device = torch.device("cuda")
modelsavedpath = './modelsaved'
EPOCH = 70
BATCHSIZE = 128
LR = 0.005
data_loader_train, data_loader_test = cifar10_data_download(128)
net = Resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.8, weight_decay=4e-4)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def train():
    best_acc = 85
    print("开始训练--网络结构:ResNet18--数据集:cifar10--学习率:0.01--momentum=0.8, weight_decay=4e-4")

    with open("log.txt", "w") as f2:
        for epoch in range(EPOCH):
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            print("\n当前Epoch:%d"%(epoch+1))
            for index,data in enumerate(data_loader_train,0):
                # 准备数据
                length = len(data_loader_train)
                inputs,labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                inputs = fast_gradient_method(net,inputs,eps=0.1,norm=np.inf)
                optimizer.zero_grad()
                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                print("===============================================")
                if predicted != labels:
                    print("FGSM attack 成功")
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[当前Epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (index + 1 + epoch * length), sum_loss / (index + 1), 100. * correct / total))
                print("===============================================")
                f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (index + 1 + epoch * length), sum_loss / (index + 1), 100. * correct / total))
                f2.write('\n')
                f2.flush()
                # print("Waiting Test!")
                # with torch.no_grad():
                #     correct = 0
                #     total = 0
                #     for data in data_loader_test:
                #         net.eval()
                #         images, labels = data
                #         images, labels = images.to(device), labels.to(device)
                #         outputs = net(images)
                #         # 取得分最高的那个类 (outputs.data的索引号)
                #         _, predicted = torch.max(outputs.data, 1)
                #         total += labels.size(0)
                #         correct += (predicted == labels).sum()
                #     print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                #     acc = 100. * correct / total
                #     # 将每次测试结果实时写入acc.txt文件中
                #
                #     f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                #     f.write('\n')
                #     f.flush()
                #     # 记录最佳测试分类准确率并写入best_acc.txt文件中
                #     if acc > best_acc:
                #         f3 = open("best_acc.txt", "w")
                #         f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                #         f3.close()
                #         best_acc = acc
            print('Saving model......')
        torch.save(net.state_dict(), '%s/resnet18_%03d_fgsm_epsilon=01.pth' % (modelsavedpath, epoch + 1))
        print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == "__main__":

    train()


