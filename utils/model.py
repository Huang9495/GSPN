import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import os
from codes.utils import make_new_state_dict

def Alexnet_reg(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.alexnet(pretrained=pretrained)
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Sequential(nn.Dropout(p=0.5),
                                                                        nn.Linear(9216,4096,True),
                                                                        nn.ReLU())),
                                                  ('fc2', nn.Sequential(nn.Dropout(p=0.5),
                                                                        nn.Linear(4096,2048,True),
                                                                        nn.ReLU())),
                                                  ('fc3', nn.Linear(2048,num_out,True))]))
    if pretrained_dir:
        if os.path.isfile(pretrained_dir):
            print("=> pretrained from '{}'".format(pretrained_dir))
            checkpoint = torch.load(pretrained_dir)
            state_dict = make_new_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
        else:
            print('pretrained file not found')

    return model

def resnet_18(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    if pretrained_dir:
        if os.path.isfile(pretrained_dir):
            print("=> pretrained from '{}'".format(pretrained_dir))
            checkpoint = torch.load(pretrained_dir)
            state_dict = make_new_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
        else:
            print('pretrained file not found')

    return model

def shufflenet(num_in, num_out,pretrained=False):
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
    model.conv1 = nn.Sequential(nn.Conv2d(num_in, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True))
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    return model

def squeezenet(num_out, pretrained=False):
    model= torchvision.models.squeezenet1_0(pretrained=pretrained)
    model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Conv2d(512, num_out, kernel_size=(1, 1), stride=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                     )
    return model

class SqueezeNet1(nn.Module):
    def __init__(self, num_in, num_out, pretrained=False):
        super(SqueezeNet1, self).__init__()
        self.num_out = num_out
        model = torchvision.models.squeezenet1_0(pretrained=pretrained)
        self.squeezenet_layer= nn.Sequential(*list(model.features.children())[1:])
        self.conv1 = nn.Conv2d(num_in, 96, kernel_size=(7, 7), stride=(2, 2))
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Conv2d(512, num_out, kernel_size=(1, 1), stride=(1, 1)),
                                         nn.ReLU(inplace=True),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                         )

    def forward(self, x):
        x = self.conv1(x)
        x = self.squeezenet_layer(x)
        x = self.classifier(x)

        return x.view(x.size(0), self.num_out)


def mnasnet(num_out, pretrained=False):
    model= torchvision.models.mnasnet0_5(pretrained=pretrained)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(in_features=1280, out_features=num_out, bias=True))
    return model


def mobilenet(num_out, pretrained=False):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(in_features=1280, out_features=num_out, bias=True))

    return model

def resnet_34(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.resnet34(pretrained=pretrained)
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    if pretrained_dir:
        if os.path.isfile(pretrained_dir):
            print("=> pretrained from '{}'".format(pretrained_dir))
            checkpoint = torch.load(pretrained_dir)
            state_dict = make_new_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
        else:
            print('pretrained file not found')

    return model


class resnet_50_joint_ls(nn.Module):
    def __init__(self, num_in, num_out, pretrained=False):
        super(resnet_50_joint_ls, self).__init__()
        self.net = torchvision.models.resnet50(pretrained= pretrained)
        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = nn.Linear(2048, num_out)
        # self.input_layer= nn.Sequential(nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #                                 nn.BatchNorm2d(64),
        #                                 nn.ReLu(inplace=True),
        #                                 nn.MaxPool2d(kernel_size=(3, 3),stride=2, padding=1, dilation=1,ceil_mode=False))

    def forward(self, input):
        output = self.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        # output = self.net.avgpool(output)
        output = output.squeeze(-1).squeeze(-1)

        output = self.fc(output)

        return output


def resnet_50(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.resnet50(pretrained=pretrained)
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    # if pretrained_dir:
    #     if os.path.isfile(pretrained_dir):
    #         print("=> pretrained from '{}'".format(pretrained_dir))
    #         checkpoint = torch.load(pretrained_dir)
    #         state_dict = make_new_state_dict(checkpoint['state_dict'])
    #         models.load_state_dict(state_dict)
    #     else:
    #         print('pretrained file not found')

    return model


def resnet_101(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.resnet101(pretrained=pretrained)
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    if pretrained_dir:
        if os.path.isfile(pretrained_dir):
            print("=> pretrained from '{}'".format(pretrained_dir))
            checkpoint = torch.load(pretrained_dir)
            state_dict = make_new_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
        else:
            print('pretrained file not found')

    return model


def resnet_152(num_out, pretrained=False, pretrained_dir=None):
    model = torchvision.models.resnet152(pretrained=pretrained)
    previous_layer_out= model.fc.weight.size()[-1]
    model.fc= nn.Linear(previous_layer_out, num_out)

    if pretrained_dir:
        if os.path.isfile(pretrained_dir):
            print("=> pretrained from '{}'".format(pretrained_dir))
            checkpoint = torch.load(pretrained_dir)
            state_dict = make_new_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
        else:
            print('pretrained file not found')

    return model

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class GoogleNet_ori(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, num_out, pretrained=True):
        super(GoogleNet_ori, self).__init__()
        base_model = torchvision.models.inception_v3(pretrained=pretrained)
        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)
        self.pos2 = nn.Linear(2048, num_out, bias=True)

    def forward(self, x):
        x = self.base_model(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(x.size(0), -1)
        pos = self.pos2(x)

        return pos


class HED_VGG16(nn.Module):
    def __init__(self):
        super(HED_VGG16, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d1 = F.interpolate(self.dsn1(conv1), size=(h, w), mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.dsn2(conv2), size=(h, w), mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.dsn3(conv3), size=(h, w), mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.dsn4(conv4), size=(h, w), mode='bilinear', align_corners=True)
        d5 = F.interpolate(self.dsn5(conv5), size=(h, w), mode='bilinear', align_corners=True)

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse

class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self,num_in, num_out, pretrained=False):
        super(GoogleNet, self).__init__()
        base_model = torchvision.models.inception_v3(pretrained=pretrained)
        self.Conv2d_1a_3x3 = BasicConv2d(num_in, 32, kernel_size=3, stride=2)

        model = []
        # models.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        # if fixed_weight:
        #     for param in self.base_model.parameters():
        #         param.requires_grad = False

        # Out 2
        self.pos2 = nn.Linear(2048, num_out, bias=True)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        # 299 x 299 x 3
        x = self.base_model(x)

        # 8 x 8 x 2048
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))

        # 1 x 1 x 2048
        x = F.dropout(x, p=0.5, training=self.training)

        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)

        # 2048
        pos = self.pos2(x)

        return pos

def load_pretrained_weights(model, pretrained_dir=0):
    if os.path.isfile(pretrained_dir):
        print("=> pretrained from '{}'".format(pretrained_dir))
        if pretrained_dir[-3:]=='tar':
            checkpoint = torch.load(pretrained_dir)
            state_dict = checkpoint['state_dict']
        else:
            state_dict = torch.load(pretrained_dir)
        if list(state_dict.keys())[1][:6]=='module':
            state_dict = make_new_state_dict(state_dict)
        model.load_state_dict(state_dict)
    else:
        print('pretrained file not found')

    return model


def load_hed_vgg16_v1(pretrained_dir):
    model = HED_VGG16_v1()
    if os.path.isfile(pretrained_dir):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained_dir)
        if list(pretrained_dict.keys())[0][:6]=='module':
            pretrained_dict = make_new_state_dict(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class HED_VGG16_v1(nn.Module):
    def __init__(self):
        super(HED_VGG16_v1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv5_1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=3, bias=True)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv5_1 = self.conv5_1(conv5)

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)
        ang = self.avg(conv5_1)
        ang = ang.squeeze(-1).squeeze(-1)
        ang = self.fc(ang)

        return d1, d2, d3, d4, d5, fuse, ang


def load_hed_vgg16_v2(pretrained_dir, num_in=3, num_out=3):
    model = HED_VGG16_v2(num_in=num_in, num_out=num_out)
    if os.path.isfile(pretrained_dir):
        model_dict = model.state_dict()
        if pretrained_dir[-3:]=='tar':
            checkpoints = torch.load(pretrained_dir)
            pretrained_dict = checkpoints['state_dict']
        else:
            pretrained_dict = torch.load(pretrained_dir)

        if list(pretrained_dict.keys())[0][:6]=='module':
            pretrained_dict = make_new_state_dict(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k[:4]!='fuse' and k[:5]!='conv1'}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class HED_VGG16_v2(nn.Module):
    def __init__(self, num_in=3, num_out=3):
        super(HED_VGG16_v2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_in, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.fuse = nn.Conv2d(3, 1, 1)
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_out, bias=True)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)

        fuse = F.sigmoid(fuse)
        ang = self.avg(conv5)
        ang = ang.squeeze(-1).squeeze(-1)
        ang = self.fc(ang)

        return d1, d2, d3, fuse, ang


class VGG16(nn.Module):
    def __init__(self, num_in, num_out):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_in, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_out, bias=True)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        ang = self.avg(conv5)
        ang = ang.squeeze(-1).squeeze(-1)
        ang = self.fc(ang)

        return ang


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            # if "fc" in name:
            #     x = x.view(x.size(0), -1)
            # x = module(x)
            # print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                x = module(x)
                print(name)
                outputs[name] = x

        return outputs





















