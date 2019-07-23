#coding:utf-8
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

def image_for_pytorch(img_file):
    Data = Image.open(img_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(224),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    imData = transform(Data)
    imData = Variable(torch.unsqueeze(imData, dim=0), requires_grad=True)
    return imData.cuda()


def fetch_model():
    resnet50_model = models.resnet50(pretrained=True)
    modules = list(resnet50_model.children())[:-1]
    resnet50_conv2 = torch.nn.Sequential(*modules)
    resnet152_conv2.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #所有求平均
    resnet50_conv2.fc = lambda x: x
    resnet50_conv2.cuda()
    return resnet50_conv2

resnet50_conv2 = fetch_model()

img_file = "test.jpg"
inputs = image_for_pytorch(img_file)
with torch.no_grad():
    features = resnet50_conv2(inputs)

"""
print(features.size())
(1, 2048, 6, 2)
"""
data=outputs.data.view(1,-1)
"""
print(data.size())
(1, 24576)
"""
data_list = data.tolist()
data_str = map(str,data_list)
data_result = ",".join(data_str)
print "%s\t%s"%(0,data_result)

