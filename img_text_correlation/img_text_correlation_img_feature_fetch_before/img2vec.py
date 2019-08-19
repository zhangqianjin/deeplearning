#coding:utf-8
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import sys


def image_for_pytorch(img_file):
    #整个图片feed到模型中
    #Data = Image.open(img_file).convert('RGB').resize((224,224),Image.ANTIALIAS)
    Data = Image.open(img_file).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,
 0.225))])
    imData = transform(Data)
    imData = torch.unsqueeze(imData, dim=0)
    return imData.cuda()
    
    def fetch_model():
    resnet_model = models.resnet34(pretrained=True)
    modules = list(resnet_model.children())[:-1]
    resnet_conv2 = torch.nn.Sequential(*modules)
    resnet_conv2.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    resnet_conv2.fc = lambda x: x
    for para in resnet_conv2.parameters():
        para.requires_grad=False
    resnet_conv2.cuda()
    return resnet_conv2

def fetch_img2id(img2id_file):
    img2id_dict = {}
    with open(img2id_file) as f:
        for line in f:
            line_list = line.strip().split("\t")
            img,id= line_list
            img2id_dict[img] = id
    return img2id_dict

if __name__ == '__main__':
   img_path = sys.argv[1]
   img2id_file = sys.argv[2]
   img2id_dict = fetch_img2id(img2id_file)
   resnet_conv2 = fetch_model()
   with torch.no_grad():
       img2id_key = img2id_dict.keys()
       for img in img2id_key:
           index = img2id_dict[img]
           img_file = "%s/%s"%(img_path,img)
           inputs = image_for_pytorch(img_file)
           outputs = resnet_conv2(inputs).squeeze()
           data_list = outputs.tolist()
           data_str = map(str,data_list)
           data_result = ",".join(data_str)
           print("%s\t%s"%(index,data_result))
