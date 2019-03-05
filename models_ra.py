#%%
###  Models
#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init
#%%
        
class Embeddings(nn.Module):
    def __init__(self, sent_size, 
                 voc_size, 
                 emb_size, 
                 dropout_rate,
                 embeddings,
                 pad_idx):
        super(Embeddings, self).__init__()
        
        self.sent_size = sent_size
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.pad_idx = pad_idx
        
        
        self.embed = nn.Embedding(self.voc_size, self.emb_size, padding_idx=self.pad_idx) 
        
        embeddings = embeddings[:self.voc_size, :self.emb_size]
        
        self.embed.weight = nn.Parameter(embeddings)
        
    def forward(self, x):
        
        o = self.embed(x)

        return o
#%%
class MySoftmax(nn.Module):
    # softmax over dim=1
    def forward(self, input_):
        batch_size = input_.size()[0]
        input_ = input_.transpose(2,1)
        output_ = torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        output_=output_.transpose(2,1)
        return output_    
#%%
class Output(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 dropout_rate,
                 mean_y):
        super(Output,self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.mean_y = mean_y
        
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.sigmoid = nn.Sigmoid()
        
        if self.mean_y != None: # for essay scoring is not none, for readability assessment it is None
            
            if mean_y.ndim == 0:
                initial_mean_value = np.expand_dims(self.mean_y, axis=1)
            
            bias_value = (np.log(initial_mean_value) - 
                          np.log(1 - initial_mean_value))
            
            
            self.linear.bias.data = torch.from_numpy(bias_value).type(torch.FloatTensor) 
         
        init.xavier_uniform(self.linear.weight)
    
    def forward(self, x):  
        
        o = self.linear(x)
        
       # o = self.sigmoid(o)    
        
        return o
    
#%%
class LSTM(nn.Module):
    def __init__(self, max_doc_len,
                 output_size, 
                 voc_size,
                 emb_size, 
                 dropout_rate,
                 utt_size,
                 embeddings):
        super(LSTM,self).__init__()
        
        self.max_doc_len = max_doc_len
        self.output_size = output_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.voc_size = voc_size
        self.num_layers = 1
        
        self.embed = embeddings
        
        #####
        self.max_sent_len = utt_size
        
        self.lstm = nn.LSTM(input_size = self.emb_size, # es
              hidden_size = self.output_size, #hs in LSTM = es to be able stack it
              num_layers = self.num_layers, 
              bidirectional = False,
              dropout = self.dropout_rate,
              batch_first=True,
              bias=True)

        init.xavier_uniform(self.lstm.weight_ih_l0)
        
        init.orthogonal(self.lstm.weight_hh_l0)
        
        init.constant(self.lstm.bias_hh_l0, 0)
        
        init.constant(self.lstm.bias_ih_l0, 0)
        
        self.tanh = nn.Tanh()
        
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def apply_mask(self, x, mask):
        
        mask = mask.unsqueeze(-1)
        
        mask = mask.expand_as(x)
        
        y = mask * x
        
        return y
        
    def forward(self, h0, c0, input_doc, seq_lens, batch_size, mask):
        #inp_doc: bs* max_doc_len
        
        doc_emb = self.embed(input_doc) #bs*max_doc_len*emb_size
        
        outputs, _ = self.lstm(doc_emb,(h0,c0))#bs*max_doc_len* lstm_out
        
        outputs = self.dropout(outputs)
        
        outputs = self.apply_mask(outputs,mask)
        
        return outputs
    
    def init_hidden(self,batch_size):
        
        hidden = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.output_size))
                
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        
        return hidden
    
#%%

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Coherence(nn.Module):
    def __init__(self, 
                 max_doc_len,
                 output_size, 
                 voc_size,
                 emb_size, 
                 dropout_rate,
                 utt_size,
                 embeddings,
                 mode,
                 table):
        super(Coherence,self).__init__()
        
        self.max_doc_len = max_doc_len
        self.output_size = output_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.voc_size = voc_size
        self.num_layers = 1
        self.mode = mode
        self.table = table
        
        self.embed = embeddings
        
        self.utt_size = utt_size
        
        self.num_group = self.max_doc_len 
        
        self.lstm = nn.LSTM(input_size = self.emb_size, # es
              hidden_size = self.output_size, #hs in LSTM = es to be able stack it
              num_layers = self.num_layers, 
              bidirectional = False,
              dropout = self.dropout_rate,
              batch_first = True,
              bias = True)

        init.xavier_uniform(self.lstm.weight_ih_l0)
        
        init.orthogonal(self.lstm.weight_hh_l0)
        
        init.constant(self.lstm.bias_hh_l0, 0)
        
        init.constant(self.lstm.bias_ih_l0, 0)
        
        self.tanh = nn.Tanh()
        
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.num_groups = self.max_doc_len 
        
        #similarity activation function
        self.get_positive = torch.nn.LeakyReLU(-1.0)
        
        # linears for sentence attention
        self.linear_1 = nn.Linear(self.output_size, int(self.output_size/2), bias=False)
        init.xavier_uniform(self.linear_1.weight)
            
        self.linear_2 = nn.Linear(int(self.output_size/2), 1, bias=False)
        init.xavier_uniform(self.linear_2.weight)
                
        self.linear_3 = nn.Linear(self.output_size, self.output_size, bias=False)
        init.xavier_uniform(self.linear_3.weight)
                
        # linear(s) for concating two h_bars
        self.linear_4 = nn.Linear(2*self.output_size, self.output_size,bias = False)
        init.xavier_uniform(self.linear_4.weight)
                
        self.softmax = MySoftmax()
        
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-08)
        
        if divmod(self.table, 1)[0] == 1.0:
            
            self.conv = nn.Conv1d(in_channels=1,
                  out_channels=1,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=True)
            
            # max pooling
            self.max_pooling = nn.MaxPool2d(kernel_size = (1,3), stride = 1)
        
       
            self.linear_5 = nn.Linear(self.num_groups-4, 100, bias = True)
            
   
        elif divmod(self.table, 1)[0] == 2.0:
            
            self.conv = nn.Conv1d(in_channels=1,
                              out_channels=100,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              groups=1,
                              bias=True)
            
            
            self.max_pooling = nn.MaxPool2d(kernel_size = (1,self.num_groups-2), stride = 1)
            
            
        elif divmod(self.table, 1)[0] == 3.0:
            
            self.conv = nn.Conv1d(in_channels=1,
                  out_channels=1,
                  kernel_size=4,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=True)
        
       
            self.linear_5 = nn.Linear(self.num_groups-3, 100, bias = True)
        
        
        else:
            
            raise NotImplemented    
        
        self.linear_6 = nn.Linear(400, 300, bias = True)
        
        self.linear_7 = nn.Linear(300, 100, bias = True)

        self.adaptive_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        self.W = nn.Parameter(torch.zeros(self.output_size,2*self.output_size,2*self.output_size))
        
        init.xavier_uniform(self.W)
        
    def apply_mask(self, x, mask):
        
        mask = mask.unsqueeze(-1)
        
        mask = mask.expand_as(x)
        
        y = mask * x
        
        return y
      
    def utt_attn(self, utt, batch_size):
        
        utt_m = self.linear_1(utt) #(bs, sent_size, 2*lstm_out)
        
        utt_m = self.tanh(utt_m)
        
        utt_attn0 = self.linear_2(utt_m) #(bs,sent_size, self.num_attn)
        
        utt_attn0= self.softmax(utt_attn0) 
        
        utt_attn  = utt_attn0.transpose(2,1)#(bs, self.num_attn, sent_size)
        
        utt_attn_lstm =  torch.bmm(utt_attn,utt) #(bs,num_attn,lstm_out)
        
        utt_attn_lstm = utt_attn_lstm.view(batch_size,-1)
        
        utt_attn_lstm = self.linear_3(utt_attn_lstm)#(bs,1,lstm_out)
        
        utt_attn_lstm = self.sigmoid(utt_attn_lstm)
        
        return utt_attn_lstm, utt_attn0
        
    
    
    def sim(self, A, B):
        # bs* utt_size, lstm_out
        
        B_trans = B.transpose(2,1)
        
        C = torch.matmul(A,B_trans)
        
        # prevent big values
        C = torch.div(C, self.output_size)
        
        C = self.get_positive(C)

        return C
    
    
    def sim_vec(self,v_i,v_j,new_version=False):
       
        if new_version==True:
            # bs*lstm_size
            sim = self.cosine_similarity(v_i,v_j)
            
            sim = sim.view(-1,1)
            
            #sim = self.get_positive(sim)
            
            return sim
        else:
            
            sim = torch.matmul(v_i, v_j.transpose(1,0))
            
            s = []
            
            for i in range(sim.size()[0]):
                
                x = sim[i,i]
                            
                x = torch.div(x, v_i.size(1))
                
                x = self.get_positive(x)
                
                s.append(x)
            
            sim = torch.stack(s,dim=0)        
     
            return sim

        
    def forward(self, h0, c0, input_doc, seq_lens, batch_size, mask):
        #inp_doc: bs* max_doc_len
        
        doc_emb = self.embed(input_doc) #bs * max_doc_len * emb_size

        h = h0

        c = c0 
        
        H = []
        
        M = []
    
        for t in range(0, self.num_group):
            
            S_t = doc_emb[:,t*self.utt_size:(t+1)*self.utt_size,:].contiguous()
            
            mask_t = mask[:,t*self.utt_size:(t+1)*self.utt_size].contiguous()
            
            if 'lstm' in self.mode:
            
                S_t_len = mask_t.sum(dim = 1)
                
                one = S_t_len - S_t_len + 1

                S_t_mask = torch.min(S_t_len,one)
                
                S_t_len = torch.max(S_t_len,one)
                            
                S_t_len_sorted, indices_sorted = torch.sort(S_t_len, descending=True) 
                
                S_t_sorted = S_t[indices_sorted]
    
                h = h.transpose(1,0)
            
                h = h[indices_sorted]
                
                h = h.transpose(1,0)
                
                
                c = c.transpose(1,0)
                
                c = c[indices_sorted]
                
                c = c.transpose(1,0)
                
                sorted_lengths = [int(item) for item in list(S_t_len_sorted.data)]
                
                S_t_packed = pack_padded_sequence(S_t_sorted,
                                                  lengths=sorted_lengths,
                                                  batch_first=True)
                
                packed_H_t, (h, c) = self.lstm(S_t_packed, (h,c))
                
                H_t, H_t_lengths = pad_packed_sequence(packed_H_t, batch_first=True)
                
                # re-arrange output, h, c to original order 
                _ , indices_orginal = torch.sort(indices_sorted)
                
                H_t = H_t[indices_orginal]
    
                h = h.transpose(1,0)
                h = h[indices_orginal]
                h = h.transpose(1,0)
               
                c = c.transpose(1,0)
                c = c[indices_orginal]
                c = c.transpose(1,0)
                
                S_t_mask = S_t_mask.view(-1,1,1)
                S_t_mask_H = S_t_mask.expand_as(H_t)          
                H_t = S_t_mask_H * H_t
                  
                S_t_mask = S_t_mask.transpose(1,0)
                S_t_mask_hid = S_t_mask.expand_as(h)
                h = S_t_mask_hid * h
                c = S_t_mask_hid * c
    
                H.append(H_t)
                
                M_t = mask_t[:,:H_t.size(1)]
              
                M.append(M_t)
                
            elif 'emb' in self.mode:
               

                H.append(S_t)
                
                M.append(mask_t)
                
            else:
                
                raise NotImplemented
            
        x_vec = []
       
        for j in range(1, len(H)):
           
            H_j = H[j]
           
            i = j - 1
           
            H_i = H[i]
           
            similarities = self.sim(H_i , H_j)
            
            similarities = self.dropout(similarities)
        
            _max, _index = F.adaptive_max_pool2d(similarities,  1, return_indices=True) #1 * bs * 1
            
            _max = _max.view(batch_size,-1)
 
            _index_i = _index[0].view(batch_size,-1)
            
            _index_j = _index[1].view(batch_size,-1)
            
            vectors = []
            
            for batch_id in range(batch_size):
                   
                    batch_id = torch.LongTensor([batch_id])
                    
                    if torch.cuda.is_available():
                    
                        batch_id = batch_id.cuda()
                    
                    max_ind_i = _index_i[batch_id].data
                    
                    max_ind_j = _index_j[batch_id].data
                    
                    h_max_i = H_i[batch_id, max_ind_i, :]
                    
                    h_max_i = h_max_i.squeeze(1) # 1, emb_size
                   
                    h_max_j = H_j[batch_id, max_ind_j, :]
                    
                    h_max_j = h_max_j.squeeze(1) # 1, emb_size
                    
                    h_ij = torch.stack([h_max_i, h_max_j], dim=1) # 1, 2, emb_size
                    
                    h_ij = torch.mean(h_ij, dim=1) # 1, emb_size

                    vectors.append(h_ij)
            
            vectors = torch.stack(vectors, dim=0)

            vectors = vectors.squeeze(1)
            
            x_vec.append(vectors)
        
        coh_output = []
        
        # compute transition over segments
        for k in range(1,len(x_vec)):
            
            x_k = x_vec[k]

            x_k_1 = x_vec[k-1]
            
            sim = self.sim_vec(x_k_1, x_k)
            
            coh_output.append(sim)

        coh_output = torch.cat(coh_output,dim=1)
        
        coh_output = self.dropout(coh_output)
        
        # learn patterns in coh_output 
        coh_output = coh_output.unsqueeze(1)
        
        coh_output = self.conv(coh_output)
        
        coh_output = self.tanh(coh_output)
        

        if divmod(self.table, 1)[0] == 1.0:
            
            coh_output = self.max_pooling(coh_output)
            
            coh_output = coh_output.squeeze(1)
            
            coh_output = self.linear_5(coh_output)
        
            coh_output = self.tanh(coh_output)
            
        elif divmod(self.table, 1)[0] == 2.0:
            
            coh_output = coh_output.squeeze(2)
            
        elif divmod(self.table, 1)[0] == 3.0:
            
             coh_output = coh_output.squeeze(1)
            
             coh_output = self.linear_5(coh_output)
             
             coh_output = self.tanh(coh_output)
             
        else :
            
            raise NotImplemented

        if 'emb' in self.mode:                    
            
            out = coh_output
            
        elif 'lstm' in self.mode:
                
            if round(divmod(self.table, 1)[1],2) == 0.0:
                
                lstm_out = torch.cat(H, dim=1)
                
                mask = torch.cat(M, dim=1)
                
                lstm_out = self.apply_mask(lstm_out, mask)
                
                lstm_out = self.dropout(lstm_out)
                
                lstm_out = lstm_out.sum(1)
                    
                seq_lens = seq_lens.unsqueeze(-1)
                    
                seq_lens = seq_lens.expand_as(lstm_out)
                    
                lstm_out = lstm_out / seq_lens
               
                out = lstm_out 
               # out = self.linear_7(lstm_out) # additive
                
               # out = F.relu(out)
                
            elif round(divmod(self.table, 1)[1],2) == 0.1:
                
                out = coh_output # that is defined over lstm outputs
            
            elif round(divmod(self.table, 1)[1],2) == 0.3:

                lstm_out = torch.cat(H, dim=1)
                
                mask = torch.cat(M, dim=1)
                
                lstm_out = self.apply_mask(lstm_out, mask)
                
                lstm_out = self.dropout(lstm_out)
                
                lstm_out = lstm_out.sum(1)
                    
                seq_lens = seq_lens.unsqueeze(-1)
                    
                seq_lens = seq_lens.expand_as(lstm_out)
                    
                lstm_out = lstm_out / seq_lens
                
                #lstm_out = self.linear_7(lstm_out) # additive
                
                #lstm_out = F.tanh(lstm_out)
                
                out = torch.cat([lstm_out,coh_output], dim=1) # coh_lstm & additive
                
                out = self.linear_6(out)
                
                out = F.tanh(out)
            
                
            else:
                
                raise NotImplemented
            
        else:
            
            raise NotImplemented
     
        return out

    
    def init_hidden(self,batch_size):
        
        hidden = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.output_size))
                
        if torch.cuda.is_available():
            
            hidden = hidden.cuda()
        
        return hidden

#%%       
   
class Model(nn.Module):
    def __init__(self,
                 max_doc_len,
                 relation_size,
                 lstm_size,
                 voc_size, 
                 emb_size,
                 dropout_rate,
                 embeddings,
                 mean_y,
                 pad_idx,
                 utt_size,
                 mode,
                 table,
                 output_layer_size):
        super(Model,self).__init__()
        
        self.max_doc_len = max_doc_len
        
        self.relation_size = relation_size
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.mean_y= mean_y
        self.embeddings = embeddings
        self.lstm_size =lstm_size
        self.mode = mode
        self.pad_idx = pad_idx
        self.utt_size  = utt_size 
        self.table = table
        self.output_layer_size = output_layer_size
        
        self.embed = Embeddings(sent_size=self.max_doc_len, 
         voc_size=self.voc_size, 
         emb_size=self.emb_size, 
         dropout_rate=self.dropout_rate,
         embeddings=embeddings,
         pad_idx = self.pad_idx)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        if self.mode == 'lstm':        
            self.lstm = LSTM(max_doc_len = self.max_doc_len,
                     output_size = lstm_size, 
                     voc_size = self.voc_size,
                     emb_size=self.emb_size, 
                     dropout_rate=self.dropout_rate,
                     utt_size = self.utt_size,
                     embeddings = self.embed)
            
            self.output = Output(input_size = lstm_size,
                     output_size=1,
                     dropout_rate=self.dropout_rate,
                     mean_y=self.mean_y)
        
        if 'coh' in self.mode:
            
            self.coh = Coherence(max_doc_len=self.max_doc_len,
                     output_size=lstm_size, 
                     voc_size=self.voc_size,
                     emb_size=self.emb_size, 
                     dropout_rate=self.dropout_rate,
                     utt_size = self.utt_size,
                     embeddings = self.embed,
                     mode = self.mode,
                     table= self.table)
            
            
            self.output = Output(input_size = 100, #in additive case is 300
                     output_size=1,
                     dropout_rate=self.dropout_rate,
                     mean_y=self.mean_y)
            
        if 'feat' in self.mode:
            
            self.normalizer = nn.BatchNorm1d(self.output_layer_size)
            
            self.feat_h1 = nn.Linear(self.output_layer_size, 100)
            
            
            
            self.output = Output(input_size = 100, #in additive case is 300
                     output_size=1,
                     dropout_rate=self.dropout_rate,
                     mean_y=self.mean_y)
            
        if ('coh' in self.mode) and ('feat' in self.mode):
           
                self.feat_h2 = nn.Linear(200, 100) # main code
                 
                self.output = Output(input_size = 100,#main code 100  #in additive case is 300
                         output_size=1,
                         dropout_rate=self.dropout_rate,
                         mean_y=self.mean_y)            
                
    def forward(self, doc, seq_lens, batch_size, mask, feat, hidden=None):
        
        if self.mode == 'lstm':
            
            batch_size = doc.size()[0]
            
            h0 = self.lstm.init_hidden(batch_size)
        
            c0 = self.lstm.init_hidden(batch_size)
            
            lstm_out = self.lstm(h0,c0, doc, seq_lens, batch_size, mask)
            
            lstm_out = lstm_out.sum(1)
            
            seq_lens=seq_lens.unsqueeze(-1)
            
            seq_lens = seq_lens.expand_as(lstm_out)
            
            lstm_out = lstm_out / seq_lens
            
            output_layer_input = lstm_out
        
        elif 'coh' in self.mode:
            
            batch_size = doc.size()[0]
            
            h0 = self.coh.init_hidden(batch_size)
        
            c0 = self.coh.init_hidden(batch_size)
            
            coh_out = self.coh(h0, c0, doc, seq_lens, batch_size, mask)
            
            output_layer_input = coh_out
            
            if 'feat' in self.mode:
                              
                feat = self.feat_h1(feat)
                
                feat = F.tanh(feat)
               
                output_layer_input = torch.cat([coh_out,feat],dim=1)

                output_layer_input = self.feat_h2(output_layer_input)
                
                output_layer_input = F.sigmoid(output_layer_input)

        elif 'feat' in self.mode:
            
            feat = self.feat_h1(feat)
            
            feat = F.tanh(feat)
            
            output_layer_input = feat
 
        score = self.output(output_layer_input)
        
        return score
