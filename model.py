import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
from model_flashinterimage import fusion_net

from RetinexFormer_arch import RetinexFormer


class final_net(nn.Module):
    def __init__(self):
        super(final_net, self).__init__()
        self.dehazing_model = fusion_net()
        self.enhancement_model =  RetinexFormer()

        try:
            self.dehazing_model.load_state_dict(torch.load(os.path.join('experiments_new2/flashinterimage/checkpoints/model', '72.pkl')), strict=True)
            print('loading dehazing_model success')
        except:
            print('loading dehazing_model error')

 

    def forward(self, hazy, scale=0.05, testing=True):

        if testing:
            
            if hazy.shape[2]==4000:
                    
                hazy1 = hazy[:, :, :, :5984]
                hazy2 = hazy[:, :, :, 16:]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)
                x = torch.cat([frame_out1[:, :, :, :16], (frame_out1[:, :, :, 16:] + frame_out2[:, :, :, :5968])/2, frame_out2[:, :, :, 5968:]], dim=-1)

                hazy1 = x[:, :, :, :3200]
                hazy2 = x[:, :, :, 2800:]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat([frame_out1[:, :, :, :2800], (frame_out1[:, :, :, 2800:] + frame_out2[:, :, :, :400])/2, frame_out2[:, :, :, 400:]], dim=-1)

                x = (x + x1 * scale ) / (1 + scale)

                    
                    
            if hazy.shape[2]==6000:
                    
                hazy1 = hazy[:, :, :5984, :]
                hazy2 = hazy[:, :, 16:, :]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)

                x = torch.cat([frame_out1[:, :, :16, :], (frame_out1[:, :, 16:, :] + frame_out2[:, :, :5968, :])/2, frame_out2[:, :, 5968:, :]], dim=-2)

                hazy1 = x[:, :, :3200, :]
                hazy2 = x[:, :, 2800:, :]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat([frame_out1[:, :, :2800, :], (frame_out1[:, :, 2800:, :] + frame_out2[:, :, :400, :])/2, frame_out2[:, :, 400:, :]], dim=-2)
                x = (x + x1 * scale ) / (1 + scale)

        else:
            with torch.no_grad():
                x = self.dehazing_model(hazy)
                
            x1 = self.enhancement_model(x)

            x = (x + x1 * scale ) / (1 + scale)
              
        return x
