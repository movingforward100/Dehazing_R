
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from saicinpainting.training.modules.ffc0 import FFCResnetBlock
from saicinpainting.training.modules.ffc0 import FFC_BN_ACT




class myFFCResblock(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=2, norm_layer=nn.BatchNorm2d,     #128--->64
                 padding_type='reflect', activation_layer=nn.ReLU,
                 resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        
        super().__init__()
        self.initial = FFC_BN_ACT(input_nc, input_nc, kernel_size=3, padding=1, dilation=1,
            norm_layer=norm_layer, activation_layer=activation_layer,
            padding_type=padding_type,
            **resnet_conv_kwargs)

        self.ffcresblock = FFCResnetBlock(input_nc, padding_type=padding_type, activation_layer=activation_layer,
            norm_layer=norm_layer, **resnet_conv_kwargs)

    

        self.final = FFC_BN_ACT(input_nc, output_nc, kernel_size=3, padding=1, dilation=1,     
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **resnet_conv_kwargs)







    def forward(self, x):

        x_l, x_g = self.initial(x)

        x_l, x_g = self.ffcresblock(x_l, x_g)
        x_l, x_g = self.ffcresblock(x_l, x_g)
        
        out_ = torch.cat([x_l, x_g], 1)

        x_lout, x_gout =self.final(out_)
        
        out = torch.cat([x_lout, x_gout], 1)
        return out



