import timm
from lib.encoder import glt_b2
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.decoder import Decoder, DASPP


class Network_UF_v2(nn.Module):
    def __init__(self, channels=128):
        super(Network_UF_v2, self).__init__()
        self.shared_encoder = glt_b2()
        pretrained_dict = torch.load('')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)

        self.DASPP = DASPP(512,channels)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, 1, 1)
        )
        self.decoder5 = Decoder(512,channels)
        self.decoder4 = Decoder(320, channels)
        self.decoder3 = Decoder(128, channels)
        self.decoder2 = Decoder(64, channels)
        self.channel = channels

    def forward(self, x):
        image = x
        en_feats = self.shared_encoder(x)
        x4, x3, x2, x1 = en_feats
        x5  = self.DASPP(x4)
        x4  = self.decoder5(x5,x4)
        x3  = self.decoder4(x4,x3)
        x2  = self.decoder3(x3,x2)
        x1  = self.decoder2(x2,x1)

        x5 = self.out(x5)
        x4 = self.out(x4)
        x3 = self.out(x3)
        x2 = self.out(x2)
        x1 = self.out(x1)

        p0 = F.interpolate(x5, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)


        return p0, f4, f3, f2, f1




