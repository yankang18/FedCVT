import torch
import torch.nn as nn

from models.resnet import resnet18


class ClientVGG8(nn.Module):

    def __init__(self, an_id, output_dim=512):
        super(ClientVGG8, self).__init__()
        self.id = str(an_id)
        print("[INFO] {0} is using ClientVGG8".format(an_id))
        act = nn.LeakyReLU
        ks = 3
        self.feature_extractor = nn.Sequential(
            # first conv block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),

            # second conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),

            # third conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=output_dim),
            act(inplace=True)
        )

    def forward(self, x):
        # print(f"x:{x.shape} {x.dtype}")
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class MyResnet18(nn.Module):

    def __init__(self, an_id, class_num=10, output_dim=512):
        super(MyResnet18, self).__init__()
        model = resnet18(pretrained=False, num_classes=class_num, output_dim=output_dim)
        self.id = str(an_id)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        # print("model.fc:", model.fc)
        self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [64, 64, 56, 56]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
