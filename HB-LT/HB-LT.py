import os
import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import Tensor
import math
import pandas as pd
from io import StringIO
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc

"""
Manually change the file name and path for your phenotype files and the haplotype files.

"""
from LSTM_IN import HaplotypeLSTM
analyzer = HaplotypeLSTM(
    phenotype_path='phenotype.txt',
    #haplotype_dir='path/to/haplotypes',
    max_blocks=500
)

analyzer.interactive_analysis()




#load the haplotype block files that previously been selected by LSTM
phenotype_data = pd.read_csv('phenotype.txt', header=None)
print(phenotype_data)

haplotypes_by_row = []

for _ in range(100): #X indicate the number of haplotype blocks that left after LSTM Apre-filtering
    haplotypes_by_row.append([])

for i in range(len(phenotype_data)):
    haplotype_file = f'filtered_haplotypes/output_{i}.txt'
    with open(haplotype_file, 'r') as file:
        patient_haplotypes = [line.strip() for line in file.readlines()]
        for row_index, haplotype in enumerate(patient_haplotypes):
            haplotypes_by_row[row_index].append([int(allele) for allele in haplotype])

print(haplotypes_by_row)

#convert the labels into label tensor
labels_tensor = torch.tensor(phenotype_data[0])

#calculate the max haplotype length for padding
second_dim_lengths=[]
for haplotype in haplotypes_by_row:
    arr = np.array(haplotype)
    second_dim_length = arr.shape[1]
    second_dim_lengths.append(second_dim_length)

block_lengths = np.array(second_dim_lengths)
print(block_lengths)

selected_tensors = [torch.tensor(rows) for rows in haplotypes_by_row]

max_rows=max(tensor.shape[0] for tensor in selected_tensors)
max_cols=max(tensor.shape[1] for tensor in selected_tensors)

padded_tensors = [torch.nn.functional.pad(tensor, (0, max_cols - tensor.shape[1], 0, max_rows - tensor.shape[0]), value=3) for tensor in selected_tensors]
padded_tensors = torch.stack(padded_tensors)
print(padded_tensors.shape)
#change the shape to [num_Patient, num_haplotype, max_length]
padded_tensors = padded_tensors.permute(1,0,2)
print(padded_tensors.shape)

padding_masks = (padded_tensors == 3)
print(padding_masks)

class MyDataset(Dataset):
    def __init__(self, padded_tensor, padding_mask, label_tensor):
        self.padded_tensor = padded_tensor
        self.padding_mask = padding_mask
        self.label_tensor = label_tensor


    def __len__(self):
        return len(self.padded_tensor)

    def __getitem__(self, index):
        return {
               'input': self.padded_tensor[index],
               'padding_mask': self.padding_mask[index],
               'label': self.label_tensor[index]}

def traintest_split(tensor,label, test_size=0.2, random_state=42):
    train_tensor, test_tensor, train_label, test_label = train_test_split(tensor,label, test_size=test_size, random_state=random_state)
    return train_tensor, test_tensor, train_label, test_label


def prepare_loader(labels_tensor, num_patients,test_size=0.2, random_state=42):
    input_tensor = padded_tensors
    padding_mask = padding_masks
    train_tensor, test_tensor, train_label, test_label = traintest_split(input_tensor, labels_tensor)
    train_padding_mask, test_padding_mask,qtrain_label, qtest_label = traintest_split(padding_mask,labels_tensor, test_size, random_state)
    train_dataset = MyDataset(train_tensor, train_padding_mask, train_label)
    test_dataset = MyDataset(test_tensor, test_padding_mask, test_label)
    train_loader = DataLoader(train_dataset, batch_size=num_patients, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=num_patients, shuffle=False)
    return train_loader, test_loader, train_tensor, train_padding_mask,train_label,test_tensor, test_padding_mask, test_label

num_patients=500
train_loader, test_loader,train_tensor, train_padding_mask,train_label, test_tensor, test_padding_mask, test_label = prepare_loader(labels_tensor, num_patients, test_size=0.2, random_state=42)

print(train_tensor.shape)
print(train_label.shape)

#positional and word embedding
vocab_size=4
embedding_dim=16
word_embedding = nn.Embedding(vocab_size, embedding_dim)#,padding_idx=3)

class PositionalEmbedding(nn.Module):
    """
    original refrence: Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Reused from https://github.com/nlpyang/hiersumm/blob/476e6bf9c716326d6e4c27d5b6878d0816893659/src/abstractive/neural.py#L38
    """

    def __init__(self, dim, max_len=5000, dropout=0.1):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEmbedding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

print(word_embedding(train_tensor).shape)
#word_embedding(train_tensor)[:2]


print(train_tensor.shape)
batch_size, n_blocks, n_tokens = train_tensor.size()

positional_embedding = PositionalEmbedding(int(embedding_dim/2), max(n_blocks,n_tokens))
word_emb = word_embedding(train_tensor)

positional_embedding.pe.size()
positional_embedding.pe[:, :n_blocks].size()
positional_embedding.pe[:, :n_tokens].size()

print('local:',positional_embedding.pe[:, :n_tokens].size())
print('inter:',positional_embedding.pe[:, :n_blocks].size())

pos_emb_local = positional_embedding.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(embedding_dim / 2))
pos_emb_inter = positional_embedding.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(embedding_dim / 2))

pos_emb_local = positional_embedding.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(embedding_dim / 2))
pos_emb_inter = positional_embedding.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(embedding_dim / 2))
comb_pos_emb = torch.cat([pos_emb_local, pos_emb_inter],-1)
#comb_pos_emb = comb_pos_emb.view(-1, max_size_of_haps,embedding_dim)
word_emb = word_emb * math.sqrt(embedding_dim) # Note that the forward function of Positional embedding is not used at all.
      #print('word_emb', word_emb.shape)
      #print(comb_pos_emb.shape)
emb = word_emb + comb_pos_emb

print(pos_emb_local.shape)
print(pos_emb_inter.shape)

#test
print(comb_pos_emb[0])
print(word_emb)
print(emb[0])

""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if(self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)  #([1600, 4, 60, 4])
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = (torch.matmul(query, key.transpose(2, 3)) + torch.matmul(key, query.transpose(2, 3)))/2
        print('query', query.shape, 'key', key.transpose(2, 3).shape)
        #print('scores', scores ==  scores.transpose(2,3))

        #scores = (query @ key.T + key @ query.T) / (2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        # exp_scores = torch.exp(scores)
        # print('exp', exp_scores.shape)
        # row_sum = exp_scores.sum(dim=-1, keepdim=True)
        # print('row', row_sum.shape)
        # col_sum = exp_scores.sum(dim=-2, keepdim=True)
        # print('col', col_sum.shape)
        # norm_factors = (row_sum + col_sum.transpose(2,3)) / 2  # Average normalization
        # norm_scores = exp_scores / norm_factors
        # print(norm_scores.shape)

        # attn = norm_scores







        attn = self.softmax(scores)
        attn = (attn + attn.transpose(2,3))/2
        #print(attn == attn.transpose(2,3))
        #print('attn',attn)
        print('attnShape', attn.shape)

        attention = attn       #.mean(1) #Here we will keep the atention for different heads sepeate and see if they are learning different things.
        print('attentionShape', attention.shape)
        #print(attention - attention.transpose(1,2))
        #print(attention.shape)
        #print(attn.shape)
        #print(value.shape)


        drop_attn = self.dropout(attn)
        if(self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, attention
        else:
            context = torch.matmul(drop_attn, value)
            return context , attention

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn


class MultiHeadedPooling(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if (use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
        self.use_final_linear = use_final_linear

    def forward(self, key, value, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x, dim=dim_per_head):
            """  projection """
            return x.view(batch_size, -1, head_count, dim) \
                .transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim)

        scores = self.linear_keys(key)
        value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)
        # key_len = key.size(2)
        # query_len = query.size(2)
        #
        # scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        if (self.use_final_linear):
            context = unshape(context).squeeze(1)
            output = self.final_linear(context)
            return output
        else:
            return context

head_count = 4
model_dim = 16

m =  MultiHeadedAttention( head_count, model_dim)
print(emb.shape)

t, w = m(emb, emb, emb) #[1600, 4, 60, 4]) The 16 is divided into 4 and 4 by function shape and the transposed. So the operations happen seperately in each head.
print(t.shape, w.shape)

import math
import torch
from torch import nn

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask = None):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        #mask = mask.unsqueeze(1)
        context, w_local = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out), w_local


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask = None):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerInterEncoder, self).__init__()
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        # src = src.view(batch_size * n_blocks, n_tokens)
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        #mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        #mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec, w_local = self.transformer_layers[i](word_vec, word_vec)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                word_vec, w_inter = self.transformer_layers[i](word_vec, batch_size, n_blocks)  # all_sents * max_tokens * dim

        word_vec = self.layer_norm(word_vec)
        #mask_hier = mask_local[:, :, None].float()
        src_features = word_vec #* mask_hier
        src_features = src_features.view(batch_size, n_blocks * n_tokens, -1)
        src_features = src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        #mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens, -1)
        #mask_hier = mask_hier.transpose(0, 1).contiguous()

        #unpadded = [torch.masked_select(src_features[:, i], mask_hier[:, i].byte()).view([-1, src_features.size(-1)])
                    #for i in range(src_features.size(1))]

        max_l = 400 #max([p.size(0) for p in unpadded])
        #mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).to(self.device)
        #mask_hier = 1 - mask_hier[:, None, :]

        #unpadded = torch.stack(
            #[torch.cat([p, torch.zeros(max_l - p.size(0), src_features.size(-1)).to(self.device)]) for p in unpadded], 1)
        src_features = src_features.mean(0)
        src_features = src_features.mean(1).unsqueeze(1)
        return src_features, w_local, w_inter


class TransformerInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.inter_att = MultiHeadedAttention(heads, self.d_per_head, dropout, use_final_linear=False)

        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        #mask_inter = mask_inter.unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        #mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks)

        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, self.heads, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)

        block_vec, w_inter = self.inter_att(block_vec, block_vec, block_vec)  # all_sents * max_tokens * dim
        block_vec = block_vec.view(batch_size, self.heads, n_blocks, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * n_blocks, self.heads * self.d_per_head)
        block_vec = self.linear(block_vec)

        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out, w_inter

num_layers = 2
d_model= 16
heads = 1
d_ff= 16
dropout= 0.1
embeddings= torch.nn.Embedding(vocab_size, embedding_dim)
inter_layers = [0]
inter_heads = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerInterEncoder(num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device)

model(train_tensor)
print(model)

#training
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

def calculate_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

num_epochs = 1500
train_losses = []
val_losses = []
train_rmses = []
val_rmses = []
block_attention_weights_list=[]
within_block_attention_list=[]
block_attention_weights_list_test=[]

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_train_loss = 0.0
    total_train_rmse = 0.0

    for data in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        inputs = data['input']
        mask = data['padding_mask']
        labels = data['label']

        output, w_within, w_inter = model(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            # snp_attention_weights_list.append(snp_attention_weights)
            block_attention_weights_list.append(w_inter)
            within_block_attention_list.append(w_within)

        total_train_loss += loss.item()
        total_train_rmse += calculate_rmse(output, labels).item()


    average_train_loss = total_train_loss / len(train_loader)
    average_train_rmse = total_train_rmse / len(train_loader)
    train_losses.append(average_train_loss)
    train_rmses.append(average_train_rmse)

    #scheduler.step(average_train_loss)

    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    total_val_rmse = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs = data['input']
            mask = data['padding_mask']
            labels = data['label']

            output, _, w_test = model(inputs)
            labels = labels.view(-1, 1)
            loss = criterion(output, labels)
            total_val_loss += loss.item()
            total_val_rmse += calculate_rmse(output, labels).item()

            if epoch % 100 == 0:
                # snp_attention_weights_list.append(snp_attention_weights)
                block_attention_weights_list_test.append(w_test)


    average_val_loss = total_val_loss / len(test_loader)
    average_val_rmse = total_val_rmse / len(test_loader)
    val_losses.append(average_val_loss)
    val_rmses.append(average_val_rmse)


len(block_attention_weights_list)

"""
The default output of HB-LT takes the mean of multi-heatmaps and is used as the output. However, the users can also choose to take the sum
or output multiple heatmaps as it is. 
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt


def cross_block_visualize(tensor, operation='none'):
    """
    Process 3D tensor of matrices and visualize results

    Args:
        tensor (torch.Tensor): Input tensor of shape [n, x, x]
        operation (str): One of 'mean', 'sum', or 'none'
    """
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
        raise ValueError("Input must be a 3D PyTorch tensor")

    n_matrices = tensor.shape[0]
    matrix_size = tensor.shape[1]

    if operation == 'mean':
        processed = tensor.mean(dim=0)
        plot_heatmap(processed, "Mean")
    elif operation == 'sum':
        processed = tensor.sum(dim=0)
        plot_heatmap(processed, "Sum")
    elif operation == 'none':
        for i in range(n_matrices):
            plot_heatmap(tensor[i], f"Heatmap {i + 1}")
    else:
        raise ValueError("Invalid operation. Choose 'mean', 'sum', or 'none'")


def plot_heatmap(matrix, title):
   
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix.detach().cpu().numpy(),
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar=False)
    plt.title(title)
    plt.show()


print("Available operations:")
print("1. Mean of matrices")
print("2. Sum of matrices")
print("3. Show all matrices")
choice = input("Enter your choice (1-3): ").strip()

operation_map = {
    '1': 'mean',
    '2': 'sum',
    '3': 'none'
}

if choice not in operation_map:
    print("Invalid choice! Using default 'none' operation")
    choice = '3'

cross_block_visualize(block_attention_weights_list[-1].squeeze(), operation_map[choice])

"""
For the within-block attention filter, the same option is also provided, mean, sum, or output multiple heatmaps as it is. 
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(matrix, title, ax=None):
    """Plot a single attention heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix,
                annot=False, 
                fmt=".2f",
                cmap="viridis",
                cbar=True,
                ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Position", fontsize=10)
    ax.set_ylabel("Position", fontsize=10)


def process_4d_attention_tensor(tensor, operation='none'):
    """
    Process 4D attention tensor and generate organized heatmaps
    Args:
        tensor (torch.Tensor): Input tensor of shape [n_blocks, n_heads, seq_len, seq_len]
        operation (str): One of 'mean', 'sum', or 'none'
    """

    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        raise ValueError("Input must be a 4D PyTorch tensor of shape [n_blocks, n_heads, seq_len, seq_len]")

    n_blocks, n_heads, seq_len, _ = tensor.shape

    if operation not in ['mean', 'sum', 'none']:
        raise ValueError("Invalid operation. Choose 'mean', 'sum', or 'none'")

    # Create subplots based on the operation
    if operation in ['mean', 'sum']:
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes] 
    else:  # 'none' operation
        fig, axes = plt.subplots(n_blocks, n_heads, figsize=(5 * n_heads, 5 * n_blocks))
        if n_blocks == 1:
            axes = [axes]  

    # Process each block
    for block_idx in range(n_blocks):
        block_tensor = tensor[block_idx] 

        if operation == 'mean':
            processed = block_tensor.mean(dim=0).detach().cpu().numpy()
            plot_attention_heatmap(processed, f"Block {block_idx + 1} Mean Attention", axes[block_idx])
        elif operation == 'sum':
            processed = block_tensor.sum(dim=0).detach().cpu().numpy()
            plot_attention_heatmap(processed, f"Block {block_idx + 1} Sum Attention", axes[block_idx])
        elif operation == 'none':
            for head_idx in range(n_heads):
                head_matrix = block_tensor[head_idx].detach().cpu().numpy()
                if n_blocks == 1:
                    ax = axes[head_idx]
                else:
                    ax = axes[block_idx, head_idx]
                plot_attention_heatmap(head_matrix, f"Block {block_idx + 1} Head {head_idx + 1}", ax)

    plt.tight_layout()
    plt.show()


# User interaction
print("Available operations:")
print("1. Mean of attention heads")
print("2. Sum of attention heads")
print("3. Show all attention heads")
choice = input("Enter your choice (1-3): ").strip()

operation_map = {
    '1': 'mean',
    '2': 'sum',
    '3': 'none'
}

if choice not in operation_map:
    print("Invalid choice! Using default 'none' operation")
    choice = '3'



def reshape_to_4d(tensor: torch.Tensor, first_dim: int) -> torch.Tensor:
    orig_dim0, orig_dim1, orig_dim2 = tensor.shape

    if orig_dim0 % first_dim != 0:
        raise ValueError(f"Cannot split dimension {orig_dim0} into {first_dim} chunks. " +
                         f"Remainder: {orig_dim0 % first_dim}")

    second_dim = orig_dim0 // first_dim

    return tensor.view(first_dim, second_dim, orig_dim1, orig_dim2)


within_attention = reshape_to_4d(within_block_attention_list[-1].squeeze(), num_block) 

print("Original shape:", original_tensor.shape)
print("New shape:", reshaped_tensor.shape)

process_4d_attention_tensor(within_attention, operation_map[choice])
