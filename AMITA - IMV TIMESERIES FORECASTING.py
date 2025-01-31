#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import random
import pandas as pd
from copy import deepcopy
from typing import Dict
import transformers
from torch import Tensor
from torch.nn import init, Parameter
import torch.nn.functional as F
import pdb
import math
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, average_precision_score
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


# In[2]:


def seed_all(seed: int = 1992):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
seed_all()


# In[4]:


class AMITA2i_LSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size,seq_len, output_dim, batch_first=True, bidirectional=True):
        super(AMITA2i_LSTM, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.initializer_range=0.02
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size,1, self.hidden_size]).float()
        self.decay_features = torch.Tensor(torch.arange(self.input_size)).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)
        self.alpha = torch.FloatTensor([0.5])
        self.imp_weight = torch.FloatTensor([0.05])
        self.alpha_imp = torch.FloatTensor([0.5])
        self.register_buffer("factor", self.alpha)
        self.register_buffer("imp_weight_freq", self.imp_weight)
        self.register_buffer("features_decay", self.decay_features)
        self.register_buffer("factor_impu", self.alpha_imp)
        
        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        
        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        
        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_f= nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        
        
        self.b_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        # Interpolation
        self.W_ht_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size,1)))
        self.W_ct_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size,1)))
        self.b_j_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        
        self.W_ht_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size,1)))
        self.W_ct_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size,1)))
        self.b_j_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size))) 
        
        #Gate Linear Unit for last records
        self.activation_layer = nn.ELU()
        self.F_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size,self.hidden_size*2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size,1)))
        self.F_beta = nn.Linear(self.seq_len, 4*self.hidden_size)
        self.layer_norm1 = nn.LayerNorm([self.input_size, self.seq_len])
        self.layer_norm = nn.LayerNorm([self.input_size, 4*self.hidden_size])
        self.Phi = nn.Linear(4*self.hidden_size, self.output_dim)
        
    @torch.jit.script_method    
    def TLSTM_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        l = last_data
        freq=freq_list
        T = self.map_elapse_time(t)
        # Short-term memory contribution
        D_ST = torch.tanh(torch.einsum("bij,ijk->bik", c_tilda_t, self.W_decomp))  
        # Apply temporal decay to D-STM
        decay_factor = torch.mul(T, self.freq_decay(freq, h_tilda_t))
        D_ST_decayed = D_ST * decay_factor
        # Long-term memory contribution
        LTM = c_tilda_t - D_ST + D_ST_decayed  
        # Combine short-term and long-term memory
        c_tilda_t = D_ST_decayed + LTM
        #frequency weights for imputation of missing data based on frequencies of features
        # Imputation gate for inputs x last records
        x_last_hidden =torch.tanh(torch.einsum("bij,ijk->bik", self.freq_decay(freq, h_tilda_t),self.W_ht_last)+\
                                  torch.einsum("bij,ijk->bik", self.freq_decay(freq, c_tilda_t),
                                  self.W_ct_last)+self.b_j_last).permute(0, 2, 1)
        
        imputat_imputs = torch.tanh(torch.einsum("bij,ijk->bik", self.freq_decay(freq, h_tilda_t), self.W_ht_mask)+\
                                    torch.einsum("bij,ijk->bik",self.freq_decay(freq, c_tilda_t),self.W_ct_mask)+\
                                    self.b_j_mask).permute(0, 2, 1)
        # Replace nan data with the impuated value generated from LSTM memory and frequencies weights

        _, x_last = self.impute_missing_data(l, freq, x_last_hidden)
        all_imputed_x, imputed_x = self.impute_missing_data(x, freq, imputat_imputs)
        
        # Ajust previous to incoporate the latest records for each feature
        last_tilda_t = self.activation_layer(torch.einsum("bij,jik->bjk", x_last, self.U_last)+self.b_last)
        h_tilda_t = h_tilda_t + last_tilda_t
        # Capturing Temporal Dependencies wrt to the previous hidden state
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +\
                               torch.einsum("bij,jik->bjk", imputed_x,self.U_j) + self.b_j)
        
        # Time Gate
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk",imputed_x, self.U_time) + 
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i= torch.sigmoid(torch.einsum("bij,jik->bjk",imputed_x, self.U_i)+\
                         torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i)+\
                         c_tilda_t*self.W_cell_i + self.b_i*self.freq_decay(freq, j_tilda_t))
        # Forget Gate
        f= torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_f)+\
                         torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f)+\
                         c_tilda_t*self.W_cell_f + self.b_f+j_tilda_t)

        f_new = f * self.map_elapse_time(t) + (1 - f) *  self.freq_decay(freq, j_tilda_t)
        # Candidate Memory Cell
        C =torch.tanh(torch.einsum("bij,jik->bjk", imputed_x, self.U_c)+\
                      torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell
        Ct = (f_new + t_gate) * c_tilda_t + i * j_tilda_t * t_gate * C
        # Output Gate        
        o = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_o)+
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o)+
                          t_gate + last_tilda_t + Ct*self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(Ct+last_tilda_t)
        
        return h_tilda_t, Ct, self.freq_decay(freq, j_tilda_t), f_new, all_imputed_x
    
    @torch.jit.script_method
    def impute_missing_data(self, x: torch.Tensor, freq_dict: torch.Tensor, x_hidden: torch.Tensor):
        # Calculate feature factor
        factor_feature = torch.div(
            torch.exp(-self.imp_weight_freq * freq_dict),
            torch.exp(-self.imp_weight_freq * freq_dict).max()).unsqueeze(1)
        
        # Calculate imputation factor
        factor_imp = torch.div(
            torch.exp(self.factor_impu * freq_dict),
            torch.exp(self.factor_impu * freq_dict).max()).unsqueeze(1)
        
        # Adjust frequencies
        frequencies = (self.seq_len-freq_dict) * torch.exp(-self.factor * self.features_decay)
        frequencies = torch.div(frequencies, frequencies.max()).unsqueeze(-1)
        
        # Compute imputed values
        imputed_missed_x = torch.where(
            factor_imp == factor_imp.max(),
            frequencies.permute(0,2,1) * x_hidden,
            factor_feature * x_hidden
        )
        
        # Replace missing values
        x_imputed = torch.where(torch.isnan(x.unsqueeze(1)), imputed_missed_x, x.unsqueeze(1))
        
        return imputed_missed_x, x_imputed
    
    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk",freq_weight.unsqueeze(-1),self.Dw)+\
                                torch.einsum("bij,ijk->bik", ht, self.W_d)+ self.b_d)
        return weights
    @torch.jit.script_method
    def forward(self, inputs, times, last_values, freqs):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
       
        seq_len = inputs.size()[0]
        imputed_inputs = torch.jit.annotate(List[Tensor], [])
        hidden_his = torch.jit.annotate(List[Tensor], [])
        weights_decay = torch.jit.annotate(List[Tensor], [])
        weights_fgate = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_hidden, prev_cell,pre_we_decay, fgate_f, imputed_x = self.TLSTM_unit(prev_hidden,
                                                                           prev_cell, 
                                                                           inputs[i],
                                                                           times[i], 
                                                                           last_values[i],
                                                                           freqs[i])
            hidden_his += [prev_hidden]
            imputed_inputs += [imputed_x]
            weights_decay += [pre_we_decay]
            weights_fgate += [fgate_f]
        imputed_inputs = torch.stack(imputed_inputs)
        hidden_his = torch.stack(hidden_his)
        weights_decay = torch.stack(weights_decay)
        weights_fgate = torch.stack(weights_fgate)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            imputed_inputs_b = torch.jit.annotate(List[Tensor], [])
            second_hidden_his = torch.jit.annotate(List[Tensor], [])
            second_weights_decay = torch.jit.annotate(List[Tensor], [])
            second_weights_fgate = torch.jit.annotate(List[Tensor], [])
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i-1]
                second_hidden, second_cell,b_we_decay,fgate_b,imputed_x_b = self.TLSTM_unit(second_hidden,
                                                                                second_cell, 
                                                                                second_inputs[i],
                                                                                time,
                                                                                last_values[i],
                                                                                freqs[i])
                second_hidden_his += [second_hidden]
                second_weights_decay += [b_we_decay]
                second_weights_fgate += [fgate_b]
                imputed_inputs_b += [imputed_x_b]
                
            imputed_inputs_b = torch.stack(imputed_inputs_b)
            second_hidden_his = torch.stack(second_hidden_his)
            second_weights_fgate = torch.stack(second_weights_fgate)
            second_weights_decay = torch.stack(second_weights_decay)
            weights_decay =torch.cat((weights_decay, second_weights_decay), dim=-1)
            weights_fgate =torch.cat((weights_fgate, second_weights_fgate), dim=-1)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
            imputed_inputs = torch.cat((imputed_inputs, imputed_inputs_b), dim=2)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=-1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
            imputed_inputs= imputed_inputs.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            weights_fgate = weights_fgate.permute(1, 0, 2, 3)
        
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha) + self.F_alpha_n_b)
        alphas = alphas.reshape(alphas.size(0), alphas.size(2),
                                alphas.size(1)*alphas.size(-1))
        mu=self.Phi(self.layer_norm(self.F_beta(self.layer_norm1(alphas))))
        out=torch.max(mu, dim=1).values
        return out, weights_decay, weights_fgate, imputed_inputs


# In[5]:


class TimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,seq_len, output_dim, dropout=0.2):
        super(TimeLSTM, self).__init__()
        # hidden dimensions
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.seq_len = seq_len
        self.output_dim= output_dim
        # Temporal embedding MWTA_LSTM
        self.amita_2i_lstm = AMITA2i_LSTM(self.input_size, self.hidden_size,
                                          self.seq_len, self.output_dim) 
    def forward(self,historic_features,timestamp, last_features, features_freqs , is_test=False):
        # Temporal features embedding
        outputs, decay_weights, fgate, imputed_inputs = self.amita_2i_lstm(historic_features,timestamp, 
                                                                           last_features, features_freqs)
        if is_test:
            return decay_weights, fgate, imputed_inputs.mean(axis=2), outputs
        else:
            return outputs


# In[6]:


class DataSampler:
    def __init__(self, percentage=0.2):
        self.percentage = percentage
        
    def mark_data_as_missing(self, data):
        # Create a copy of the data to avoid modifying the original tensor
        data_with_missing = data.clone()

        # Identify the observed (non-NaN) values
        observed_mask = ~torch.isnan(data)
        observed_flat_indices = torch.where(observed_mask.view(-1))[0]

        # Randomly sample a percentage of the observed data indices to mark as missing
        num_samples = int(self.percentage * observed_flat_indices.size(0))
        sampled_indices = observed_flat_indices[torch.randperm(observed_flat_indices.size(0))[:num_samples]]

        # Convert flat indices to 3D indices for the original shape of the data
        sampled_3d_indices = np.unravel_index(sampled_indices.cpu().numpy(), data.shape)

        # Mark the selected observed data points as missing (NaN) in the copy of the data
        selected_data = data[sampled_3d_indices]
        data_with_missing[sampled_3d_indices] = torch.tensor(float('nan'))
        
        # Return the modified data with additional missing values
        return selected_data, data_with_missing, sampled_3d_indices


# In[7]:


class EnhancedLossCalculator:
    def __init__(self):
        pass
    
    def dynamic_weighted_loss(self, sampled_data, sampled_imputed_x, data_freqs, outputs, labels, criterion):
        # Calculate mean absolute error (MAE)
        loss_imp = torch.mean(torch.abs(sampled_data - sampled_imputed_x))
        
        # Normalize frequencies
        normalized_freqs = data_freqs / torch.max(data_freqs)
        
        # Apply dynamic weighting using exponential decay
        dynamic_weights = torch.exp(-0.005 * (1 - normalized_freqs))
        
        # Apply weighted loss calculation
        weighted_loss_imp = loss_imp * dynamic_weights
        weighted_loss_imp = torch.sum(weighted_loss_imp) / torch.sum(dynamic_weights)
        
        # Calculate prediction loss
        prediction_loss = criterion(outputs.squeeze(-1), labels.squeeze(-1))
        
        # Combine imputation loss and prediction loss
        total_loss = prediction_loss + weighted_loss_imp
        return weighted_loss_imp, total_loss


# In[8]:


class EarlyStopping:
    def __init__(self, mode, path, patience=3, delta=0):
        if mode not in {'min', 'max'}:
            raise ValueError("Argument mode must be one of 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("Argument patience must be a positive integer.")
        if delta < 0:
            raise ValueError("Argument delta must not be a negative number.")

        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0

    def _is_improvement(self, val_score):
        """Return True iff val_score is better than self.best_score."""
        if self.mode == 'max' and val_score > self.best_score + self.delta:
            return True
        elif self.mode == 'min' and val_score < self.best_score - self.delta:
            return True
        return False

    def __call__(self, val_score, model):
        """
        Return True iff self.counter >= self.patience.
        """

        if self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("Val loss improved, Saving model's best weights.")
            return False
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Stopped early. Best val loss: {self.best_score:.4f}')
                return True


class TrainerHelpers:
    def __init__(self, input_dim, hidden_dim, seq_length, output_dim, device, optim, loss_criterion, schedulers, num_epochs, patience_n=50, task=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.device = device
        self.optim = optim
        self.loss_criterion = loss_criterion
        self.schedulers = schedulers
        self.num_epochs = num_epochs
        self.patience_n = patience_n
        self.task = task

    @staticmethod
    def acc(predicted, label):
        predicted = predicted.sigmoid()
        pred = torch.round(predicted.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train_model(self, model, train_dataloader):
        model.train()
        running_loss, running_corrects, mae_train = 0.0, 0, 0
        for bi, inputs in enumerate(tqdm(train_dataloader, total=len(train_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            # Sampling observed datapoints to random missing 
            sampled_data,data_with_missing, indices = data_sampler.mark_data_as_missing(temporal_features)
            self.optim.zero_grad()
            _, _,imputed_inputs, outputs = model(data_with_missing, timestamp,
                                                 last_data, data_freqs, is_test=True)
            sampled_imputed_x=imputed_inputs[indices]
            sampled_freqs=(seq_length-data_freqs[indices])
        
            if self.task:
                loss_imp,loss = loss_calculator.dynamic_weighted_loss(sampled_data, sampled_imputed_x,
                                                                      sampled_freqs, outputs.sigmoid(),
                                                                      labels, self.loss_criterion)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                mae_train += loss_imp.item()
                running_corrects += self.acc(outputs,labels)
            else:
                loss_imp,loss = loss_calculator.dynamic_weighted_loss(sampled_data, sampled_imputed_x,
                                                                      sampled_freqs, outputs,labels,
                                                                      self.loss_criterion)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                mae_train += loss_imp.item()
        if self.task:
            epoch_loss = running_loss / len(train_dataloader)
            epoch_mae_imp = mae_train / len(train_dataloader)
            epoch_acc = running_corrects / len(train_dataloader.dataset)
            return epoch_mae_imp, epoch_loss, epoch_acc
        else:
            epoch_loss = running_loss / len(train_dataloader)
            epoch_mae_imp = mae_train / len(train_dataloader)
            return epoch_mae_imp, epoch_loss

    def valid_model(self, model, valid_dataloader):
        model.eval()
        running_loss, running_corrects, mae_val = 0.0, 0, 0
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(valid_dataloader, total=len(valid_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs,_,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            sampled_data,data_with_missing, indices = data_sampler.mark_data_as_missing(temporal_features)
            with torch.no_grad():
                _, _,imputed_inputs,outputs = model(data_with_missing, timestamp,
                                                    last_data, data_freqs, is_test=True)
            sampled_imputed_x=imputed_inputs[indices]
            sampled_freqs=(seq_length-data_freqs[indices])
            
            if self.task:
                loss_imp,loss = loss_calculator.dynamic_weighted_loss(sampled_data, sampled_imputed_x,
                                                                      sampled_freqs, outputs.sigmoid(),
                                                                      labels, self.loss_criterion)
                running_loss += loss.item()
                mae_val += loss_imp.item()
                running_corrects += self.acc(outputs, labels)
            else:
                loss_imp,loss = loss_calculator.dynamic_weighted_loss(sampled_data, sampled_imputed_x,
                                                                      sampled_freqs, outputs, 
                                                                      labels,self.loss_criterion)
                mae_val += loss_imp.item()
                running_loss += loss.item()
            fin_targets.append(labels.cpu().detach().numpy())
            fin_outputs.append(outputs.cpu().detach().numpy())
        if self.task:
            epoch_mae_imp = mae_val / len(valid_dataloader)
            epoch_loss = running_loss / len(valid_dataloader)
            epoch_acc = running_corrects / len(valid_dataloader.dataset)
            return epoch_mae_imp, epoch_loss, epoch_acc, np.vstack(fin_targets), np.vstack(fin_outputs)
        else:
            epoch_mae_imp = mae_val / len(valid_dataloader)
            epoch_loss = running_loss / len(valid_dataloader)
            mse = mean_squared_error(np.vstack(fin_targets), np.vstack(fin_outputs))
            mae = mean_absolute_error(np.vstack(fin_targets), np.vstack(fin_outputs))
            return epoch_mae_imp, epoch_loss , mse, mae, np.vstack(fin_targets), np.vstack(fin_outputs)

    def eval_model(self, model_class, model_path,  test_dataloader):
        # Initialize the model architecture
        model = model_class(self.input_dim, self.hidden_dim, self.seq_length, self.output_dim).to(self.device)
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode
        model.eval()
        fin_targets, fin_outputs = [], []
        fin_inputs_i, inputs_outputs_i = [], []
        all_decays, fgate_weights = [], []
        for bi, inputs in enumerate(tqdm(test_dataloader, total=len(test_dataloader), leave=False, 
                                         desc='Evaluating on test data')):
            temporal_features, timestamp, last_data, data_freqs,_,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            # Random mask observed values 
            sampled_data,data_with_missing, indices = data_sampler.mark_data_as_missing(temporal_features)
            with torch.no_grad():
                 _, _,imputed_inputs,outputs= model(data_with_missing, timestamp,
                                                    last_data, data_freqs, is_test=True)
            sampled_imputed_x=imputed_inputs[indices]
            sampled_freqs=(seq_length-data_freqs[indices])
            if self.task:
                fin_outputs.append(outputs.sigmoid().cpu().detach().numpy())
                
            else:
                fin_outputs.append(outputs.cpu().detach().numpy())
            fin_targets.append(labels.cpu().detach().numpy())
            fin_inputs_i.append(sampled_data.cpu().detach().numpy())
            inputs_outputs_i.append(sampled_imputed_x.cpu().detach().numpy())
        return fin_inputs_i, inputs_outputs_i, all_decays, fgate_weights, np.vstack(fin_targets), np.vstack(fin_outputs)

    def train_validate_evaluate(self,model_class, model, model_name, train_loader, val_loader, test_loader, params, model_path):
        best_losses, all_scores = [], []
        es = EarlyStopping(mode='min', path=f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                           patience=self.patience_n)
        for epoch in range(self.num_epochs):
            if self.task:
                loss_imp, loss, accuracy = self.train_model(model, train_loader)
                eval_loss_imp, eval_loss, eval_accuracy, __, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step()
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss imp: {loss_imp:.8f}, train loss: {loss:.8f}, accuracy: {accuracy:.8f} | valid loss imp: {eval_loss_imp:.8f}, valid loss: {eval_loss:.8f}, accuracy: {eval_accuracy:.4f}")
                if es(eval_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
            else:
                loss_imp, loss = self.train_model(model, train_loader)
                eval_loss_imp, eval_loss, mse_loss, mae_loss, _, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step()
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss imp: {loss_imp:.8f}, train loss: {loss:.8f} | valid loss imp: {eval_loss_imp:.8f} valid loss: {eval_loss:.8f} valid mse loss: {mse_loss:.8f}, valid mae loss: {mae_loss:.8f}")
                if es(mse_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
        if self.task:
            _, _, _, y_true, y_pred = self.valid_model(model, val_loader)
            pr_score = average_precision_score(y_true, y_pred)
            print(f"[INFO] PR-AUC ON FOLD :{model_name} -  score val data: {pr_score:.4f}")
        else:
            _, _, _, _, y_true, y_pred = self.valid_model(model, val_loader)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(
                f"[INFO] mse loss & mae loss on validation data Fold {model_name}: mse loss: {mse:.8f} - mae loss: {mae:.8f}")
        if self.task:
            f1_scores_folds = []
            targets, outputs, real_inputs, imputed_inputs = self._evaluate_model(model_class, 
                                                                                 f"{os.path.join(model_path,f'model_{model_name}.pth')}",
                                                                                 test_loader)
            scores = self.metrics_binary(targets, outputs)
            scores_imp = self.metrics_reg_imp(real_inputs, imputed_inputs)
            delta, f1_scr = self.best_threshold(np.vstack(targets), np.vstack(outputs))
            f1_scores_folds.append((delta, f1_scr))
            all_scores.append([scores, f1_scores_folds,scores_imp])
            print(f"[INFO] Results on test Folds {all_scores}")
        else:
            targets, outputs, real_inputs, imputed_inputs = self._evaluate_model(model_class, 
                                                                                 f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                                                                                 test_loader)
            scores = self.metrics_reg(targets, outputs, params)
            scores_imps = self.metrics_reg_imp(imputed_inputs, real_inputs)
            scores_imp = self.metrics_reg_imp(real_inputs, imputed_inputs)
            all_scores.append([scores, scores_imp, scores_imps])
            np.savez(os.path.join(model_path, f"test_data_fold_{model_name}.npz"), 
                      reg_scores=scores,imput_scores=scores_imp, true_labels=targets, 
                      imputs_scores=scores_imps,predicted_labels= outputs, 
                     real_x = real_inputs, imputed_x=imputed_inputs)
            print(f"[INFO] Results on test Folds {all_scores}")
        return all_scores

    def _evaluate_model(self, model_class, model_path,  test_dataloader):
        targets, predicted = [], []
        real_inputs, imputed_inputs, all_decays, fgate_weights = [], [], [], []
        fin_inputs_i, inputs_outputs_i, _, _, y_pred, y_true = self.eval_model(model_class, model_path,
                                                                               test_dataloader)
        targets.append(y_true)
        predicted.append(y_pred)
        imputed_inputs.append(inputs_outputs_i)
        real_inputs.append(fin_inputs_i)
        targets_all = [np.vstack(targets[i]) for i in range(len(targets))]
        predicted_all = [np.vstack(predicted[i]) for i in range(len(predicted))]
        
        real_inputs_ =[np.hstack(real_inputs[i]) for i in range(len(real_inputs))]
        imputed_inputs_ =[np.hstack(imputed_inputs[i]) for i in range(len(imputed_inputs))]
        return targets_all, predicted_all, real_inputs_, imputed_inputs_

    @staticmethod
    def metrics_binary(targets, predicted):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_true)
            auc_score = metrics.auc(fpr, tpr)
            pr_score = metrics.average_precision_score(y_pred, y_true)
            scores.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return scores

    @staticmethod
    def best_threshold(train_preds,y_true):
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(y_true, np.array(train_preds) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray, rowcount: np.int64, featurecount: np.int64):
        return 1 - (1 - r2_score(actual, predicted)) * (rowcount - 1) / (rowcount - featurecount)

    def metrics_reg(self, targets, predicted, rescale_params):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            target_max, target_min = rescale_params['data_targets_max'], rescale_params['data_targets_min']
            targets_y_true = y_true * (target_max - target_min) + target_min
            targets_y_pred = y_pred * (target_max - target_min) + target_min
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            n = y_true.shape[0]
            r2 = r2_score(targets_y_true, targets_y_pred)
            adj_r2 = self.adjusted_r2(targets_y_true, targets_y_pred, n, self.input_dim)
            scores.append([rmse, mae, r2, adj_r2])
        return scores
   
    def metrics_reg_imp(self, real, imputed):
        scores = []
        for y_true, y_pred, in zip(real, imputed):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            n = y_true.shape[0]
            adj_r2 = self.adjusted_r2(y_true, y_pred, n, self.input_dim)
            scores.append([rmse, mae, r2, adj_r2])
        return scores


# dn="/media/sangaria/8TB-FOLDERS/PAPER_REVIEWS_DATASETS_TASKS/TIMESERIES/24_HRS_DATA/ItaLyAirQuality"
# task_dataset ="ItaLyAirQuality_24_HRS_DATA"

# In[12]:


dn="/media/sangaria/8TB-FOLDERS/PAPER_REVIEWS_DATASETS_TASKS/TIMESERIES/AMITA2i/24_HRS_DATA/BEIJINGAIRQUALITY MULTISITE"
task_dataset ="BEIJINGAIRQUALITY_24_HRS_DATA_128"

dn="/media/sangaria/8TB-FOLDERS/PAPER_REVIEWS_DATASETS_TASKS/TIMESERIES/AMITA2i/48_HRS_DATA/ETT_H1"
task_dataset ="ETT_H1_48_BATCH_SIZE_128"
# In[ ]:


dn="/media/sangaria/8TB-FOLDERS/PAPER_REVIEWS_DATASETS_TASKS/TIMESERIES/AMITA2i/24_HRS_DATA/ITALYAIRQUALITY_POWERTRANSFORMER_24_HRS_DATA"
task_dataset ="ITALYAIRQUALITY_24_POWERTRANSFORMER"


# dn="/media/sangaria/8TB-FOLDERS/PAPER_REVIEWS_DATASETS_TASKS/TIMESERIES/AMITA2i/24_HRS_DATA/IHEPC_FORCASTING_24"
# task_dataset ="IHEPC_FORCASTING_24_BATCH_SIZE_128"

# In[13]:


all_dataset_loader = np.load(os.path.join(os.path.join(dn,task_dataset),
                                          "train_test_data.npz"), 
                                          allow_pickle=True)


# In[14]:


train_val_loader = all_dataset_loader['folds_data_train_valid']


# In[15]:


test_loader = all_dataset_loader['folds_data_test']


# In[16]:


all_dataset_loader.files


# In[17]:


dataset_settings = np.load(os.path.join(os.path.join(dn,task_dataset),
                                        "data_max_min.npz"), 
                                         allow_pickle=True)


# In[18]:


dataset_settings.files


# In[19]:


dataset_settings['data_targets_max'], dataset_settings['data_targets_min']


# In[20]:


data_max, data_min= dataset_settings['data_max'], dataset_settings['data_min']


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## device= 'cuda:1' 
seq_length = dataset_settings['seq_length'].item()
input_dim = dataset_settings['input_dim'].item()
hidden_dim, output_dim  = 128, 1#dataset_settings['output_size'].item()
seq_length, input_dim, hidden_dim, output_dim


# In[22]:


LEARNING_RATE = 1e-3
optimizer_config={"lr": 1e-3, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}
NUM_EPOCHS =500
NUM_FOLDS = 10
model_name="AMITA2i".lower()
n_patience = 100
batch_size=64
steps_per_epoch = int(dataset_settings['shape_data'][0] / batch_size / NUM_FOLDS)
total_steps_per_fold = int(steps_per_epoch * NUM_EPOCHS)
num_warmup_steps = int(0.1 * total_steps_per_fold)
num_warmup_steps,total_steps_per_fold


# In[23]:


amita2i = TimeLSTM(input_dim, hidden_dim, seq_length, output_dim).to(device)
# Create an instance of the class
data_sampler = DataSampler(percentage=0.3)
loss_calculator = EnhancedLossCalculator()
optimizer = torch.optim.Adam(amita2i.parameters(), **optimizer_config)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                               num_warmup_steps=num_warmup_steps, 
                                                               num_training_steps=total_steps_per_fold, 
                                                               num_cycles=2)
criterion = nn.MSELoss().to(device)
best_model_wts = deepcopy(amita2i.state_dict())
amita2i


# In[24]:


train_valid_inference = TrainerHelpers(input_dim, hidden_dim, seq_length, output_dim,
                                       device, optimizer, criterion, scheduler, NUM_EPOCHS, 
                                       patience_n=n_patience, task=False)
train_valid_inference


# In[25]:


task_dataset.split("_")[0], task_dataset


# In[26]:


main_path = f"/home/sangaria/Videos/Second journal paper/REVIEWS/AMITA2i-LSTM/LOSS FUNCTION/IMPUTATION/NEW_FUNCS/{task_dataset.split('_')[0]}_BENCHMARKS_RESULTS/30_M_RATE"
task_path=f"{os.path.join(main_path, f'{task_dataset}')}"
if not os.path.exists(task_path):
    os.makedirs(task_path)
task_path


# In[27]:


scores_folds= []
for idx, (train_loader, test_data) in enumerate(zip(train_val_loader ,test_loader)):
    print(f'[INFO]: Training on fold : {idx+1}')
    # Reset the model weights
    amita2i.load_state_dict(best_model_wts)
    train_data, valid_data= train_loader
    scores= train_valid_inference.train_validate_evaluate(TimeLSTM, amita2i, idx+1,train_data,valid_data,
                                                          test_data, dataset_settings, task_path)
    scores_folds.append(scores)


# In[ ]:


mse_mae_r2_task =np.mean([fold[0][0] for fold in scores_folds], axis=0)
mse_mae_r2_task_imp = np.mean([fold[0][1] for fold in scores_folds], axis=0)


# In[ ]:


mse_mae_r2_task, mse_mae_r2_task_imp


# In[ ]:


np.savez(os.path.join(task_path, f"test_data_fold_{task_dataset.split('_')[0]}_results.npz".lower()), 
                      reg_scores=scores_folds ,
                      task_results=mse_mae_r2_task,
                      imputation_results= mse_mae_r2_task_imp)


# In[ ]:




