import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.resnet18 import  Resnet18
from torchvision import transforms
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Resnet18().to(device)
net.load_state_dict(torch.load("./modelsaved/resnet18_050.pth"))
class SaveOutput:
    def __init__(self):
        self.outputs = []
        self.inputs  = []

    def __call__(self, module,module_in,module_out):
        # print(module)
        self.inputs.append(module_in)
        self.outputs.append(module_out)
    def clear(self):
        self.outputs = []
        self.inputs = []
save_output = SaveOutput()
hook_handles = []
for layer in net.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
img = cv2.imread("./dog.png")
totensor= transforms.Compose([
        transforms.ToTensor(),
        # 归一化处理
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
x = totensor(img).unsqueeze(0).to(device)
out = net(x)

print(save_output.outputs[0][0].size())
# picture = (save_output.outputs[0][0]).cpu().detach().numpy() # 图像数据 64*32*32
# # 通道1下的数据即 32*32
# print(picture[1])
# print(picture[1].shape)
# print(picture[1].max(),picture[1].min())


