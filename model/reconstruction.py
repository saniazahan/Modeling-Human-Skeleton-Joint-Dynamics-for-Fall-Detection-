#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:24:50 2021

@author: 22905553
"""
import torch
import torch.nn as nn    

class SpatialGraphConv_Recon(nn.Module):

    def __init__(self, in_channels, out_channels, k_num,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_outpadding=0, t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels*k_num,
                                         kernel_size=(t_kernel_size, 1),
                                         padding=(t_padding, 0),
                                         output_padding=(t_outpadding, 0),
                                         stride=(t_stride, 1),
                                         dilation=(t_dilation, 1),
                                         bias=bias)

    def forward(self, x, A):

        x = self.deconv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num,:,:,:]
        
        #print(x1.shape)
        #print(x2.shape)        
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        #print(x1.shape)
        #x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        

        return x1.contiguous()

class StgcnReconBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 output_padding=(1,0),
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True,
                 activation='relu'):
        super().__init__()
        assert len(dilation) == 2
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)
        

        self.gcn_recon = SpatialGraphConv_Recon(in_channels=in_channels,
                                         out_channels=out_channels,
                                         k_num=kernel_size[1],
                                         t_dilation = dilation[0],
                                         t_kernel_size=t_kernel_size)
        self.tcn_recon = nn.Sequential(nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels=out_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=(kernel_size[0], 1),
                                                          stride=(stride, 1),
                                                          padding=padding,
                                                          dilation = dilation,
                                                          output_padding=output_padding),
                                       nn.BatchNorm2d(out_channels),
                                       nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=1,
                                                             stride=(stride, 1),
                                                             output_padding=(stride-1,0)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x, A):
        
        res = self.residual(x)
        x = self.gcn_recon(x, A)        
        x = self.tcn_recon(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x

        return x
    
if __name__ == "__main__":
    # For debugging purposes
    import sys
    import random
    sys.path.append('..')
    
    N, C, T, V = 12, 384, 13, 25
    x = torch.randn(N,C,T,V)
    recon_layer_6 = StgcnReconBlock(128+3, 30, (1, 2), dilation = (1,1), output_padding=(0,0), stride=1)
    p = recon_layer_6(x)
