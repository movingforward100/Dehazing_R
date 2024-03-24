import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re

from test_dataset_for_testing_nosplit import dehaze_test_dataset
from model import final_net


#os.chdir

parser = argparse.ArgumentParser(description='Dehazing_R')
parser.add_argument('--test_dir', type=str, default='data/test/')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()

output_dir =args.output_dir
if not os.path.exists(output_dir + '/'):
    os.makedirs(output_dir + '/', exist_ok=True)
    
test_dir = args.test_dir
test_batch_size = args.test_batch_size

test_dataset = dehaze_test_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)


device = 'cuda:0'
print(device)

MyEnsembleNet= final_net()


# --- Load the network weight --- #
try:
   MyEnsembleNet.load_state_dict(torch.load(os.path.join('weights/Dehazing_R_checkpoint', 'model.pkl')))
   print('--- MyEnsembleNet loaded ---')
except:
   print('--- no weight loaded ---')

MyEnsembleNet = MyEnsembleNet.to(device)



with torch.no_grad():
    for batch_idx, (hazy,name) in enumerate(test_loader):

        hazy = hazy.to(device)
            

        frame_out = MyEnsembleNet(hazy, testing=True)
                
        frame_out=frame_out.to(device)
        fourth_channel=torch.ones([frame_out.shape[0],1,frame_out.shape[2],frame_out.shape[3]],device='cuda:0')
     
        frame_out_rgba=torch.cat([frame_out,fourth_channel],1)
        print(frame_out_rgba.shape)
        
        name= re.findall("\d+",str(name))
        imwrite(frame_out_rgba, output_dir + '/' + str(name[0])+'.png', range=(0, 1))
      












