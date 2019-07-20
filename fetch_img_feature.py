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
        transforms.Resize(256),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    imData = transform(Data)
    imData = Variable(torch.unsqueeze(imData, dim=0), requires_grad=True)
    return imData.cuda()

resnet152_model = models.resnet152(pretrained=True)
modules = list(resnet152_model.children())[:-1]
resnet152_conv2 = torch.nn.Sequential(*modules)
resnet152_conv2.fc = lambda x: x
resnet152_conv2.cuda()

img_file = "test.jpg"
inputs = image_for_pytorch(img_file)
with torch.no_grad():
    outputs = resnet152_conv2(inputs)

outputs = outputs[:,:,0,0]
print(outputs.data.shape)
data=outputs.data.view(1,-1)
data_arr = data.cpu().numpy()[0]
data_list =  list(data_arr)
data_str = map(str,data_list)
data_result = ",".join(data_str)
print "%s\t%s"%(0,data_result)

