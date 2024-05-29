import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from myFFCResblock0 import myFFCResblock

from FlashInternImage.models.flash_intern_image  import FlashInternImage


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2   
    x02 = x[:, :, 1::2, :] / 2   
    x1 = x01[:, :, :, 0::2]    
    x2 = x02[:, :, :, 0::2]       
    x3 = x01[:, :, :, 1::2]      
    x4 = x02[:, :, :, 1::2]  
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)

class DWT_transform(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency,dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency,dwt_high_frequency

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

class dwt_ffc_UNet2(nn.Module):
    def __init__(self,output_nc=3, nf=16):
        super(dwt_ffc_UNet2, self).__init__()
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(16, nf-1, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8-8, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8-16, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*4, nf*4, name, transposed=False, bn=False, relu=False, dropout=False)#有改动

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16+16, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16+8, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8+4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4+2, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2+1, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.initial_conv=nn.Conv2d(3,16,3,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.layer1 = layer1
        self.DWT_down_0= DWT_transform(3,1)
        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)
        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)
        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)
        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.bn2=nn.BatchNorm2d(32)
        self.tail_conv2 = nn.Conv2d(nf*2, output_nc, 3,padding=1, bias=True)


        self.FFCResNet = myFFCResblock(input_nc=64, output_nc=64)

    def forward(self, x):
        conv_start=self.initial_conv(x)
        conv_start=self.bn1(conv_start)
        conv_out1 = self.layer1(conv_start)
        dwt_low_0,dwt_high_0=self.DWT_down_0(x)
        out1=torch.cat([conv_out1, dwt_low_0], 1)
        conv_out2 = self.layer2(out1)
        dwt_low_1,dwt_high_1= self.DWT_down_1(out1)
        out2 = torch.cat([conv_out2, dwt_low_1], 1)
        conv_out3 = self.layer3(out2)

        dwt_low_2,dwt_high_2 = self.DWT_down_2(out2)
        out3 = torch.cat([conv_out3, dwt_low_2], 1)

        # conv_out4 = self.layer4(out3)
        # dwt_low_3,dwt_high_3 = self.DWT_down_3(out3)
        # out4 = torch.cat([conv_out4, dwt_low_3], 1)

        # conv_out5 = self.layer5(out4)
        # dwt_low_4,dwt_high_4 = self.DWT_down_4(out4)
        # out5 = torch.cat([conv_out5, dwt_low_4], 1)

        # out6 = self.layer6(out5)
   
        
        out3_ffc= self.FFCResNet(out3)


        dout3 = self.dlayer6(out3_ffc)

        # Tout6_out5 = torch.cat([dout6, out5, dwt_high_4], 1)


        # Tout5 = self.dlayer5(Tout6_out5)
        # Tout5_out4 = torch.cat([Tout5, out4,dwt_high_3], 1)
        # Tout4 = self.dlayer4(Tout5_out4)
        # Tout4_out3 = torch.cat([Tout4, out3,dwt_high_2], 1)        # Tout3 = self.dlayer3(Tout4_out3)
        Tout3_out2 = torch.cat([dout3, out2,dwt_high_1], 1)
        Tout2 = self.dlayer2(Tout3_out2)
        Tout2_out1 = torch.cat([Tout2, out1,dwt_high_0], 1)
        Tout1 = self.dlayer1(Tout2_out1)
        
        Tout1_outinit = torch.cat([Tout1, conv_start], 1)
        tail1=self.tail_conv1(Tout1_outinit)
        tail2=self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)


        return dout1




class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class knowledge_adaptation_convnext(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = FlashInternImage()

        checkpoint_model = torch.load('flash_intern_image_l_22kto1k_384.pth')['model']
        checkpoint_model_keys = list(checkpoint_model.keys())
        #print(checkpoint_model_keys)
        #print(len(checkpoint_model_keys))

        for k in checkpoint_model_keys:
            if 'conv_head' in k:
                del checkpoint_model[k]
                print(k)
            elif k in ['head.weight', 'head.bias']:
                del checkpoint_model[k]
                print(k)
            else:
                continue

        params = self.encoder.state_dict() 
        ss = []

        for k, v in params.items():
            ss.append(k)
   

        try:
            self.encoder.load_state_dict(checkpoint_model, strict=True)
            print('Loading FlashInterimage success')
        except:
            print('Loading FlashInterimage error')

        self.up_block= nn.PixelShuffle(2)
        self.up_block2= nn.PixelShuffle(4)
        self.attention0 = CP_Attention_block(default_conv,1280, 3)
        self.attention1 = CP_Attention_block(default_conv, 320, 3)
        self.attention2 = CP_Attention_block(default_conv, 240, 3)
        self.attention3 = CP_Attention_block(default_conv, 140, 3)
        self.attention4 = CP_Attention_block(default_conv, 32, 3)
        self.attention5 = CP_Attention_block(default_conv, 8, 3)
        self.conv_ = nn.Conv2d(140, 128, kernel_size=3, padding=1)
        self.conv_process_1 = nn.Conv2d(8, 8, kernel_size=3,padding=1)
        self.conv_process_2 = nn.Conv2d(8, 8, kernel_size=3,padding=1)

    
    def forward(self, input):
        x_layer1, x_layer2, x_layer3 = self.encoder(input)

        #print(x_layer1.shape)
        #print(x_layer2.shape)
        #print(x_layer3.shape)

        x_mid = self.attention0(x_layer3)    
        
        x = self.up_block(x_mid)     
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1) 

        x = self.up_block(x)           
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), 1)    

        x = self.up_block(x)           
        x = self.attention3(x)     

        x = self.conv_(x)             

        x = self.up_block(x)           
        x = self.attention4(x)

        x = self.up_block(x)            #[8, 384, 384]
        x = self.attention5(x)    

        x=self.conv_process_1(x)
        out=self.conv_process_2(x)
        return out


class fusion_net(nn.Module):
    def __init__(self):
        super(fusion_net, self).__init__()
        self.dwt_branch=dwt_ffc_UNet2()
        self.knowledge_adaptation_branch=knowledge_adaptation_convnext()
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(11, 3, kernel_size=7, padding=0), nn.Tanh())
    def forward(self, input):
        dwt_branch=self.dwt_branch(input)
        knowledge_adaptation_branch=self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
