# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Fall_detection/
import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.autograd import Variable
from utils import import_class, count_params
#from model.reconstruction import StgcnReconBlock

from model.adjGraph import adjGraph
from model.part_attention import part_attention

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x
    
class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x
    
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 max_frame,
                 graph_args,
                 keep_prob = 0.9,
                 block_size = 41,
                 in_channels=3):
        super(Model, self).__init__()

        self.graph_hop = adjGraph(**graph_args)
        A = torch.tensor(self.graph_hop.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.num_class =  num_class
        
        #self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384
        bias = True
        
        self.joint_embed = embed(3, 3, norm=True, bias=bias)
        self.dif_embed = embed(3, 3, norm=True, bias=bias) #601
        
        self.fc = nn.Linear(c3, num_class)
        #self.fc_part = nn.Linear(c3, num_class)
        
        self.part_att = part_attention( 
                    self.A,
                    keep_prob,
                    block_size,    
                    max_frame,      
                    num_class = num_class,
                    ) 
    
    def motion_descriptor(self, x):
        ''' This module captures the motion of and velocity of consecutive frames'''
        
        motion = []
        for i in range(x.shape[2]-2):
            pt0 = x[:,:,i,:]
            pt1 = x[:,:,i+1,:]
            pt2 = x[:,:,i+2,:]
            if i-1 != 0:
                pt_1 = x[:,:,i-1,:]
            else:
                pt_1 = 0
            if i-2 != 0:
                pt_2 = x[:,:,i-2,:]
            else:
                pt_2 = 0        
            dp1 = pt1 - pt_1
            dp2 = pt2 + pt_2 - 2*pt0
            
            m = pt0 + dp1 + dp2
            
            motion.append(m)
        #motion.append(m)
        #motion.append(m)
        motion = torch.stack(motion)

        #motion = motion.reshape(x.shape)
        return motion
    
    
    def forward(self, x, target_pred=False):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        '''
        dd = torch.isnan(x)                
        if True in dd:
                    print('Problem in x')
        '''
        #x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()    # N, C, T, V
        '''
        mot = self.motion_descriptor(x)        
        input = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T        
        mot = mot.permute(1, 2, 3, 0).contiguous()  # N, C, V, T
        #print(mot.shape)
        mot = torch.cat([mot.new(N*M, mot.size(1), V, 2).zero_(), mot], dim=-1)
        #print(mot.shape)
        pos = self.joint_embed(input)        
        mot = self.dif_embed(mot)
        dy = pos + mot
        dy = dy.permute(0,1,3,2).contiguous() # N, C, T, V
        '''
        # Dynamic Representation        
        pos = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        dif = pos[:, :, :, 1:] - pos[:, :, :, 0:-1] #  
        dif = torch.cat([dif.new(N*M, dif.size(1), V, 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(pos)        
        dif = self.dif_embed(dif)
        dy = pos + dif
        dy = dy.permute(0,1,3,2).contiguous() # N, C, T, V        
        
    
        # Apply part based convolution
        part_out, pred = self.part_att(dy)
        #print(part_out.shape)
        '''
        if torch.is_tensor(target_pred):
            #target_motion = target_pred[:, :, 1:, :] - target_pred[:, :, 0:-1, :] #  #self.motion_descriptor(target_pred)
            #predicted_motion = pred[:, :, 1:, :] - pred[:, :, 0:-1, :]# self.motion_descriptor(pred)
            target_motion = self.motion_descriptor(target_pred)
            predicted_motion = self.motion_descriptor(pred)
        else:
        '''
        target_motion = False
        predicted_motion = False
        
        #print(part_out.shape)
        out = F.relu(part_out)
        
        out_channels = out.size(1)
        out = out.reshape(N, M, out_channels, -1)   
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence
        
        out = self.fc(out)       

        return out, predicted_motion, target_motion


if __name__ == "__main__":
    # For debugging purposes
    import sys
    import random
    sys.path.append('..')
    from thop import profile
    from thop import clever_format
    from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
    
    N, C, T, V, M = 6, 3, 250, 25, 2
    x = torch.randn(N,C,T,V,M)
    target = torch.randn(N*M,C,10,V)
    random.seed(33)
    model = Model(
        num_class=2,
        num_point=25,
        num_person=2, 
        max_frame=250,
        graph_args = {'layout': 'ntu-rgb+d','strategy': 'spatial','max_hop':4}
    )
    model.eval()

   
    out, pred, target = model.forward(x, target)
    
    macs, params = profile(model, inputs=(x, target))
    macs, params = clever_format([macs, params], "%.3f")
    
    print('Model total # params:', count_params(model)) #4231550
    
    flops = FlopCountAnalysis(model, x.float())
    print(flop_count_table(flops, max_depth=4))
    print(flop_count_str(flops))
    print(flops.total())
    

    for child in list(model.children()):        
        if isinstance(child, nn.Linear)==False:
            print(child)        
            for param in list(child.parameters()):            
                print(param.requires_grad )
                
                param.requires_grad = False
    
    child_counter = 0
    for child in list(model.children()): 
        #print("child ",child_counter)
        if child_counter == 3:
            
            for children_of_child in child.children():
                grand_child_counter = 0
                for grand_child in children_of_child.children():
                    
                    if grand_child_counter != 2:
                        #print("grand_child_counter ",grand_child_counter)
                        #print(grand_child)
                        for param in grand_child.parameters():
                            param.requires_grad = False
                    grand_child_counter += 1
        else:
            if isinstance(child, nn.Linear)==True:
                for param in list(child.parameters()):            
                #print(param.requires_grad )
                    param.requires_grad = True
            else:
                for param in list(child.parameters()):            
                #print(param.requires_grad )
                    param.requires_grad = False    
        
        child_counter += 1
        
    for child in list(model.children()):                     
            for name, param in list(child.named_parameters()):            
                print(name, param.requires_grad)
                #print(param.requires_grad )