from torchvision import models
from torch import nn

def VGG19_bn(input_nc,fc_out):
    model_ft = models.vgg19_bn(pretrained = True)
    model_ft.features[0] = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
    num_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_features,fc_out)

    return model_ft

def Resnet18(input_nc,out_fc):
    model_ft = models.resnet18(pretrained = True)
    model_ft.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,out_fc)

    return model_ft

def Alexnet(input_nc,out_fc):
    model_ft = models.alexnet(pretrained = True)
    model_ft.features[0] = nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2)
    num_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_features,out_fc)
    return model_ft

'''
model = Alexnet(4,2)
print(model)
'''
