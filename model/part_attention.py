# cd /home/uniwa/students3/students/22905553/linux/phd_codes/My_part_based_convolution/
import torch
import torch.nn as nn
from model.adjGraph import adjGraph
import numpy as np
from model.reconstruction import StgcnReconBlock
import math
# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv_part(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv_part, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A, index):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        #print(x.shape)
        #print(A[:self.s_kernel_size].shape)
        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctv', (x, A[:self.s_kernel_size,index,:])).contiguous()
      
        return x

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        #print(x.shape)
        #print(A[:self.s_kernel_size].shape)
        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size,:,:])).contiguous()
      
        return x
    
#    p = torch.einsum('nkctv,kvw->nctw', (xx, A[:,index,index])).contiguous()
    
class Spatial_Block_part(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, **kwargs):
        super(Spatial_Block_part, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv_part(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Define proportion or neurons to dropout
        #self.dropout = nn.Dropout(0.20)

    def forward(self, x, A, index):

        res_block = self.residual(x)

        x = self.conv(x, A, index)
        x = self.bn(x)
        #x = self.dropout(self.relu(x + res_block))
        x = self.relu(x + res_block)
        return x
    
class Spatial_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, max_frame, residual=False, **kwargs):
        super(Spatial_Block, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            #self.residual = lambda x: x
            self.residual = nn.Sequential(
                #nn.Conv2d(in_channels, out_channels, 1),
                nn.AdaptiveMaxPool2d((max_frame, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.AdaptiveMaxPool2d((max_frame, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Define proportion or neurons to dropout
        #self.dropout = nn.Dropout(0.20)

    def forward(self, x, A):

        res_block = self.residual(x)
        #print(res_block.shape)
        x = self.conv(x, A)
        x = self.bn(x)
        #x = self.dropout(self.relu(x + res_block))
        #print(x.shape)
        x = self.relu(x + res_block)

        return x
    
class Temporal_Block(nn.Module):
    def __init__(self, in_channel, out_channel, temporal_window_size, keep_prob, block_size, stride=1, residual=False, **kwargs):
        super(Temporal_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)
        self.keep_prob = keep_prob
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, (stride,1)),
                nn.AdaptiveMaxPool2d((1, 25)),
                nn.BatchNorm2d(out_channel),
            )

        self.conv = nn.Conv2d(in_channel, out_channel, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        ## Dropgraphs
        #self.dropS = DropBlock_Ske()
        #self.dropT = DropBlockT_1d(block_size=block_size)
        
        # Define proportion or neurons to dropout
        #self.dropout = nn.Dropout(0.20)

    def forward(self, x, A, index = False):

        res_block = self.residual(x)
        #print(res_block.shape)
        x = self.conv(x)
        x = self.bn(x)
        #print(x.shape)
        #if isinstance(index,(list, np.ndarray)):
           # x = self.dropT(self.dropS(x, self.keep_prob, A[:,index,:], x.shape[3]), self.keep_prob)
        #else:
           # x = self.dropT(self.dropS(x, self.keep_prob, A[:,:,:], x.shape[3]), self.keep_prob)
        #print(x.shape)
        #print("end")
        #x = self.dropout(self.relu(x + res_block))
        x = self.relu(x + res_block)
        return x

class cnn(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, padding=0, bias = True):
        super(cnn, self).__init__()
        self.padding = (padding,0)
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=(2,1), stride = (2,1), padding = self.padding, bias=bias)
        self.bn = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.dropout(self.bn(self.cnn(x)))
        return self.relu(x)
    
class Basic_Block(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, temporal_window_size, keep_prob, block_size, stride=1, max_frame=300, residual=False, **kwargs):
        super(Basic_Block, self).__init__()
        self.cnn = cnn(in_channel, out_channel, padding=(max_frame%2))
        self.sgcn = Spatial_Block(in_channel, in_channel, max_graph_distance, max_frame, residual)
        self.tcn = Temporal_Block(in_channel, out_channel, temporal_window_size, keep_prob, block_size, stride, residual)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        cnn = self.cnn(x)
        
        gcn = self.tcn(self.sgcn(x, A), A)
        
        return self.relu(cnn+gcn)
    
class Basic_Block_rec(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, temporal_window_size, keep_prob, block_size, stride=1, max_frame=300, residual=False, **kwargs):
        super(Basic_Block_rec, self).__init__()
        self.sgcn = Spatial_Block(in_channel, in_channel, max_graph_distance, max_frame, residual)
        self.tcn = Temporal_Block(in_channel, out_channel, temporal_window_size, keep_prob, block_size, stride, residual)
        
    def forward(self, x, A):
        conv = self.tcn(self.sgcn(x, A), A)
       
        return conv
        
class Basic_Block_part(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, temporal_window_size, keep_prob, block_size, stride=1, residual=False, **kwargs):
        super(Basic_Block_part, self).__init__()
        self.parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4, 21]) - 1          # torso
            ]
        self.res = Basic_Block(in_channel, out_channel, max_graph_distance, temporal_window_size, keep_prob, block_size, stride-1, residual)
        self.sgcn =  Spatial_Block_part(in_channel, out_channel, max_graph_distance, residual)
        self.tcn = Temporal_Block(out_channel, out_channel, temporal_window_size, keep_prob, block_size, stride-1, residual)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        res = self.res (x, A)
        #print(res.shape)
        part_x = [x[:,:,:,self.parts[i]] for i in range(len(self.parts))]
            
        
        # Apply activation to the sum of the pathways
        part_out = []
        for i, part in enumerate(part_x):
            p = self.tcn(self.sgcn(part, A, self.parts[i]), A, self.parts[i])
            #p = F.relu(self.sgcn1(part) + self.gcn3d1(part), inplace=True)
            #p = self.tcn1(p)
            part_out.append(p)
            #print(p.shape)
            #break
        part_out = torch.cat(part_out,axis=3)
        #print(part_out.shape)
        part_out = self.relu(part_out+res)
        #print(part_out.shape)
        return part_out

class part_attention(nn.Module):
    def __init__(self, 
                 A,
                 keep_prob,
                 block_size,
                 max_frame,
                 num_class,
                 kernel_size=[9,2],
                 edge_importance_weighting=True):
        super(part_attention, self).__init__()
        self.keep_prob = keep_prob
        self.A = A
        temporal_window_size, max_graph_distance = kernel_size
        spatial_kernel_size = self.A.size(0)
        
        
        in_channels = 3
        num_point = 25
        stride = 2
        
        # channels
        c1 = 96
        c2 = c1*2   #192
        c3 = c2*2   #384
    
        
        self.networks = nn.ModuleList((
            #Basic_Block_part(in_channels, c1, max_graph_distance, temporal_window_size, keep_prob, block_size, stride, residual=True),
            Basic_Block(in_channels, c1, max_graph_distance, temporal_window_size, keep_prob, block_size, stride, max_frame, residual=True),
            Basic_Block(c1, c2, max_graph_distance, temporal_window_size, keep_prob, block_size, stride, math.ceil(max_frame/2), residual=True),
            Basic_Block(c2, c3, max_graph_distance, temporal_window_size, keep_prob, block_size, stride, math.ceil(max_frame/4), residual=True)
        ))
        '''
        self.reconstruction_net = nn.ModuleList((
            Basic_Block_rec(c3, c3, max_graph_distance, temporal_window_size, keep_prob, block_size, stride+5, residual=False),
            Basic_Block_rec(c3, c2, max_graph_distance, temporal_window_size, keep_prob, block_size, stride+12, residual=False),            
            StgcnReconBlock(c2+3, 30, (1, spatial_kernel_size), dilation = (1,1), output_padding=(0,0), stride=1, padding=False)
        ))
        '''
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.networks
            ])
            '''
            self.edge_importance_recons = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.reconstruction_net
            ])
            '''
        else:
            self.edge_importance = [1] * len(self.networks)
            #self.edge_importance_recons = [1] * len(self.reconstruction_net)
            
            
    def forward(self, x):  
        
        data_last = x[:,:,-11:-10,:].clone()
        #print(data_last.shape)
        # forwad
        Adj = self.A.to(x.device) 
        for gcn, importance in zip(self.networks, self.edge_importance):
            x = gcn(x, Adj* importance)
            #print(x.shape)
        #print(x.shape)
        #print('rec')
        
        pred = x
        '''
        for i, (gcn, importance) in enumerate(zip(self.reconstruction_net, self.edge_importance_recons)):
            if i==2:
                pred = gcn(torch.cat((pred,data_last),1), Adj* importance)
            else:
                pred = gcn(pred, Adj* importance)
            #
        pred = pred.contiguous().view(-1, 3, 10, 25)  
        '''
        #print(pred.shape)
        return x, pred
    
