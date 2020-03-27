#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Long Short Term Memory unit (LSTM), Gated Recurrent Unit (GRU) ###
# From https://github.com/emadRad/lstm-gru-pytorch 

import math
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

# Recurrent Cells

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.Gh2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Gx2h = nn.Linear(input_size, hidden_size, bias=bias)

        self.Ih2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Ix2h = nn.Linear(input_size, hidden_size, bias=bias)

        self.Fh2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Fx2h = nn.Linear(input_size, hidden_size, bias=bias)

        self.Oh2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Ox2h = nn.Linear(input_size, hidden_size, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
            
        gy = self.tanh(self.Gx2h(input) + self.Gh2h(hx[0]))
        iy = self.sigmoid(self.Ix2h(input) + self.Ih2h(hx[0]))
        fy = self.sigmoid(self.Fx2h(input) + self.Fh2h(hx[0]))
        cy = fy*hx[1] + iy*gy
        oy = self.sigmoid(self.Ox2h(input) + self.Oh2h(hx[0]))
        hy = oy*self.tanh(cy)

        return (hy, cy)

class BasicRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(BasicRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity=='tanh':
          self.nonlinearity = nn.Tanh()
        elif nonlinearity=='relu':
          self.nonlinearity = nn.ReLU()
        else:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        hy = self.nonlinearity(self.x2h(input) + self.h2h(hx))

        return hy

    
    
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.Zx2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.Zh2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.Rx2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.Rh2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.Gx2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.Gh2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()
        

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        zy = self.sigmoid(self.Zx2h(input) + self.Zh2h(hx))
        ry = self.sigmoid(self.Rx2h(input) + self.Rh2h(hx))
        gy = self.tanh(self.Gx2h(input) + self.Gh2h(hx))
        hy = zy*hx + (1 - zy)*gy

        return hy


# Recurrent Models

class RNNModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(RNNModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.rnn_cell_list = nn.ModuleList()
        
        if mode == 'LSTM':
            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
        elif mode == 'GRU':
            self.rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))

        elif mode == 'RNN_TANH':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))

        elif mode == 'RNN_RELU':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid RNN mode selected.")


        self.att_fc = nn.Linear(self.hidden_size, 1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        
    def forward(self, input, hx=None):
        outs = []

        # Input shape : Batch_Size x Seq_size x Input_size
        tmp_input = input.transpose(1,0)

        for layer in self.rnn_cell_list:
            outs = []
            for Xt in tmp_input:
              # Xt shape : Batch_size x Input_size
              outs.append(layer(Xt, hx))
              hx = outs[-1]
            if self.mode=='LSTM':
              tmp_input = [o[0] for o in outs]
            else:
              tmp_input = outs
            hx = None

        if self.mode=='LSTM':
          out = outs[-1][0].squeeze()
        else:
          out = outs[-1].squeeze()

        out = self.fc(out)
        
        return out
    

class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.rnn_cell_list = nn.ModuleList()
        
        if mode == 'LSTM':
            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
                
            self.backward_rnn_cell_list.append(LSTMCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.backward_rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))

        elif mode == 'GRU':
            self.rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
                
            self.backward_rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.backward_rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))

        elif mode == 'RNN_TANH':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))
            
            self.backward_rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.backward_rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))

        elif mode == 'RNN_RELU':
            self.rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
                
            self.backward_rnn_cell_list.append(BasicRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.backward_rnn_cell_list.append(BasicRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid RNN mode selected.")


        self.att_fc = nn.Linear(2*self.hidden_size, 1) # 2*hidden_size because we have 2 outputs from the 2 RNNs
        self.fc = nn.Linear(2*self.hidden_size, self.output_size)
  
        
    def forward(self, input, hx=None):
        outs = []
        outs_rev = []
        
        if hx is None:
          h0, hT = None, None
        else:
          h0, hT = hx

        tmp_input = input.transpose(1,0)

        for layer in self.rnn_cell_list:
            outs = []
            for Xt in tmp_input:
              # Xt shape : Batch_size x Input_size
              outs.append(layer(Xt, h0))
              h0 = outs[-1]

            if self.mode=='LSTM':
              tmp_input = [o[0] for o in outs]
            else:
              tmp_input = outs

            h0 = None

        tmp_input = input.transpose(1,0)

        for layer in self.backward_rnn_cell_list:
            outs_rev = []
            for Xt in reversed(tmp_input):
              # Xt shape : Batch_size x Input_size
              outs_rev.append(layer(Xt, hT))
              hT = outs_rev[-1]

            if self.mode=='LSTM':
              tmp_input = [o[0] for o in outs_rev]
            else:
              tmp_input = outs_rev
            
            hT = None

        if self.mode=='LSTM':
          out = outs[-1][0].squeeze() 
          out_rev = outs_rev[0][0].squeeze()
        else:
          out = outs[-1].squeeze()
          out_rev = outs_rev[0].squeeze()

        out = torch.cat((out, out_rev), 1)

        out = self.fc(out)
        return out


