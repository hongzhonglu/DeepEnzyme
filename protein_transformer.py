import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].cuda()
        #return self.dropout(x)
        #return F.dropout(x, dropout, training=self.training)
        '''xx = torch.mean(torch.squeeze(x), 1).tolist()
        max_att = max([float(att) for att in xx])
        att = ['%.4f' % (float(att) / max_att) for att in xx]
        with open('/home/lu2021/DeepEnyme/Code/analysis/PafA_att.txt', 'w') as output:
            for i in att:
                output.write(i + '\n')'''
        return x


class TransformerBlock(nn.Module):
    def __init__(self, nhead, dropout, d_model, hid_size, layers_trans, max_len):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(hid_size, nhead, hid_size*4, dropout=dropout)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers_trans)
        self.PositionalEncoding = PositionalEncoding(max_len, hid_size)
        self.linear = nn.Linear(hid_size, d_model)

    def forward(self, word_vectors):
        word_vectors = self.PositionalEncoding(word_vectors)
        word_vectors = word_vectors.permute(1, 0, 2)

        word_vectors = self.transformer_encoder(word_vectors)

        word_vectors = word_vectors.permute(1, 0, 2)
        out = self.linear(word_vectors)
        out = out.squeeze(0)
        return out
