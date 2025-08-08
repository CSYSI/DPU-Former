import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from einops import rearrange
import numbers


class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.sa(x)
        y = x*out
        return y
class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)+self.mp(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class joint_attention(nn.Module):
    def __init__(self, channels):
        super(joint_attention, self).__init__()
        self.CA = CA()
        self.SA = SA(channels)

    def forward(self, x):
        x_res = x
        x = self.CA(x)
        x = self.SA(x)
        return x+x_res

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class Cross_Attention(nn.Module): # Interactive cross-attention (ICA)
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1conv_1_1 = nn.Conv2d(dim,dim,kernel_size=1)

        self.qkv1conv_3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.Local = Local(dim)
        self.FFT   = FFT_fusion(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.batch = nn.BatchNorm2d(dim)
        self.relu  = nn.ReLU(True)
    def forward(self, x1, x2):
        b, c, h, w = x2.shape
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        q_1 = self.qkv1conv_3_1(self.qkv1conv_1_1(x1))
        k_1 = self.qkv2conv_3_1(self.qkv1conv_1_1(x1))
        v_1 = self.qkv3conv_3_1(self.qkv1conv_1_1(x1))

        q_2 = self.qkv1conv_3_2(self.qkv1conv_1_1(x2))
        k_2 = self.qkv2conv_3_2(self.qkv1conv_1_1(x2))
        v_2 = self.qkv3conv_3_2(self.qkv1conv_1_1(x2))

        q_1 = rearrange(q_1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_1 = rearrange(k_1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_1 = rearrange(v_1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_2 = rearrange(q_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_2 = rearrange(k_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_2 = rearrange(v_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_1 = torch.nn.functional.normalize(q_1, dim=-1)
        k_1 = torch.nn.functional.normalize(k_1, dim=-1)
        q_2 = torch.nn.functional.normalize(q_2, dim=-1)
        k_2 = torch.nn.functional.normalize(k_2, dim=-1)

        atten_1 = (q_1 @ k_1.transpose(-2, -1)) * self.temperature
        atten_2 = (q_2 @ k_2.transpose(-2, -1)) * self.temperature

        atten_1 = atten_1.softmax(dim=-1)
        atten_2 = atten_2.softmax(dim=-1)
        out_1 = atten_1 @ v_2
        out_2 = atten_2 @ v_1
        out_1 = rearrange(out_1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_2 = rearrange(out_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_g = out_2 + out_1

        out_l = self.Local(x1)+self.Local(x2)
        out = self.FFT(out_g,out_l)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class Local(nn.Module): # Local perspective
    def __init__(self, dim):
        super().__init__()
        self.pconv0   = nn.Conv2d(dim,dim,1,bias=True)
        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv7 = nn.Conv2d(dim, dim, 7, 1, 3, bias=True, groups=dim)
        self.out     = nn.Conv2d(dim,dim,1,bias=True)

    def forward(self, x):
        x_3 = self.dwconv3(self.pconv0(x))
        x_5 = self.dwconv5(self.pconv0(x)+x_3)
        x_7 = self.dwconv7(self.pconv0(x)+x_5)
        x   =self.out(torch.add(torch.add(x_3,x_5),x_7))
        return x

class FFT_fusion(nn.Module): # Fourier-space merging strategy (FMS)
    def __init__(self, dim):
        super().__init__()
        self.weight   = nn.Sequential(
            nn.Conv2d(dim,dim//32,1,bias=True),
            nn.BatchNorm2d(dim//32),
            nn.ReLU(True),
            nn.Conv2d(dim//32, dim, 1, bias=True),
            nn.Sigmoid())

    def forward(self, x_T, x_C):
        _, _, H, W = x_T.shape
        x_T_f = torch.fft.rfft2(x_T.float(), norm='ortho')
        x_C_f = torch.fft.rfft2(x_C.float(), norm='ortho')
        T_weight = self.weight(x_T_f.real)
        C_weight = self.weight(x_C_f.real)
        G_weight = T_weight+C_weight
        x_T_f = G_weight * x_T_f + x_T_f
        x_C_f = G_weight * x_C_f + x_C_f
        x_F      = torch.add(x_T_f,x_C_f)
        x = torch.fft.irfft2(x_F, s=(H,W), norm='ortho')
        return x

class Module1_F(nn.Module):
    def __init__(self, dim=128, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(Module1_F, self).__init__()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias)
    def forward(self, x1, x2):
        x_g = self.attn(self.norm1(x1),self.norm1(x2))
        x_res = torch.add(x1,x2)
        x   = self.project_out(x_g) + x_res
        return x

class MRA(nn.Module): # Structural enhancement module (SEM)
    def __init__(self, out_channels):
        super(MRA, self).__init__()

        self.conv1= nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),)
        self.conv0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,dilation=1,padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),)

        self.AConv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=3,dilation=3), nn.BatchNorm2d(out_channels),nn.ReLU(True),)
        self.AConv5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5,dilation=5), nn.BatchNorm2d(out_channels),nn.ReLU(True),)
        self.AConv7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=7,dilation=7), nn.BatchNorm2d(out_channels),nn.ReLU(True),)

        self.out = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x1 = self.conv1(x)
        x0 = self.conv0(x1)
        x3 = self.AConv3(x1)
        x5 = self.AConv5(x1)
        x7 = self.AConv7(x1)
        x_r_0 = -1 * (torch.sigmoid(x0)) + 1
        x_r_3 = -1 * (torch.sigmoid(x3)) + 1
        x_r_5 = -1 * (torch.sigmoid(x5)) + 1
        x_r_7 = -1 * (torch.sigmoid(x7)) + 1
        attention_r = x_r_3+x_r_5+x_r_7+x_r_0
        x_r = attention_r.mul(x1)
        x   = self.out(torch.cat((x1,x_r),1)) + x

        return x


class Decoder(nn.Module):  #DPU-Former decoder
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.conv_x0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

        self.Joint_attention = joint_attention(out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.fusion = Module1_F(out_channels)

        self.out = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

        self.MRA = MRA(out_channels)

    def forward(self, x0, x):
        x = self.Joint_attention(self.conv_x(x))
        x0 = self.Joint_attention(self.conv_x0(x0))
        x0 = F.interpolate(x0, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_f  = self.fusion(x0, x)
        x_r = self.MRA(x_f)
        x_res = self.conv1(x)
        x = self.out(torch.cat([x_f,x_res,x_r], dim=1)) + x

        return x


class DASPP(nn.Module):
    def __init__(self, inchannel, outchannel=128):

        super(DASPP, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),nn.BatchNorm2d(outchannel),nn.ReLU(True)
        )
        self.branch_main_mp = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1), nn.BatchNorm2d(outchannel), nn.ReLU(True)
        )
        self.branch0 = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),nn.BatchNorm2d(outchannel),nn.ReLU(True))
        self.branch1 = nn.Sequential(nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=3,dilation=3), nn.BatchNorm2d(outchannel), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(outchannel*2, outchannel, kernel_size=3, stride=1, padding=6,dilation=6),nn.BatchNorm2d(outchannel),nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(outchannel*3, outchannel, kernel_size=3, stride=1, padding=12,dilation=12),nn.BatchNorm2d(outchannel),nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(outchannel*4, outchannel, kernel_size=3, stride=1, padding=18, dilation=18),nn.BatchNorm2d(outchannel),nn.ReLU(True))
        self.branch5 = nn.Sequential(nn.Conv2d(outchannel * 5, outchannel, kernel_size=3, stride=1, padding=24, dilation=24),nn.BatchNorm2d(outchannel), nn.ReLU(True))
        self.out = nn.Sequential(
            nn.Conv2d(outchannel * 6, outchannel, kernel_size=1),nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel , outchannel, kernel_size=3, padding=1), nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, kernel_size=1), nn.BatchNorm2d(outchannel), nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)+self.branch_main_mp(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        x = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(torch.cat((x,branch1),1))
        branch3 = self.branch3(torch.cat((x,branch1,branch2),1))
        branch4 = self.branch4(torch.cat((x,branch1,branch2,branch3),1))
        branch5 = self.branch5(torch.cat((x,branch1,branch2,branch3,branch4),1))
        out = self.out(torch.cat([branch_main,branch1, branch2, branch3, branch4,branch5],1)) + x
        return out















