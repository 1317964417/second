import torch
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from model.resnet18 import  Resnet18

from Randomly_divide_the_dataset import newData

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Resnet18().to(device)
# net.load_state_dict(torch.load("./modelsaved/resnet18_050.pth"))
net.load_state_dict(torch.load("./modelsaved/resnet18_050_fgsm_epsilon=01.pth"))
net.eval()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_loader = newData(BATCHSIZE=200)

list = []
for param in net.named_parameters():
    print(param[0])

def hook(module,input,output):
    # print(module)
    print(output.data.shape)
    list.append(output.data.tolist())


handle = net.layer3[1].conv1.register_forward_hook(hook)

with torch.no_grad():
    for index,data in enumerate(data_loader,0):
        inputs,_ = data
        inputs = inputs.to(device)
        result = net(inputs)

handle.remove()
print(len(list[0]))
print(len(list[0][0]))
data1 = list[0] # 200*64*32*32
total = 0
total_list =[]
for cols in range(len(data1[0][0])):
    for rows in range(len(data1[0][0])):
        for channels in range(len(data1[0])):
            for block in range(len(data1)):
                total += data1[block][channels][rows][cols]
        total_list.append(total)
        total = 0


print(total_list)
test = pd.DataFrame(data=total_list)
test.to_csv('./layersaved/truck/layer3[1]conv1_fgsm.csv', encoding='gbk')
plt.title("Sum of corresponding neuron values in all channels")
plt.xlabel("Neuron ordinal number")
plt.ylabel("The sum of the values of the neurons on all channels")
plt.plot(total_list)
plt.show()


















