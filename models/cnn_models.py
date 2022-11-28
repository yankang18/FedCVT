import torch
import torch.nn as nn


class ClientVGG8(nn.Module):

    def __init__(self, an_id):
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
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            # second conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            # third conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(ks, ks), stride=(1, 1), padding=1),
            act(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=48),
            act(inplace=True)
        )

    def forward(self, x):
        # print(f"x:{x.shape} {x.dtype}")
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
