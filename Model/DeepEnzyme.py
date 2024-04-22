import torch
from torch import nn
from Code.Model.GCN import GCN
from Code.Model.protein_transformer import TransformerBlock as protein_transformer
import torch.nn.functional as F


class DeepEnzyme(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size, layers_trans):
        super(DeepEnzyme, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_wordGCN = nn.Embedding(n_word, dim)
        self.embed_wordTrans = nn.Embedding(n_word, hid_size)
        self.W_out = nn.ModuleList([nn.Linear(3 * dim, 3 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(3 * dim, 1)
        self.gcn = GCN(dim, hidden_dim1, hidden_dim2, dim, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.smiles_transformer = protein_transformer(nhead, dropout, dim, hid_size, layers_trans, max_len=n_fingerprint)
        self.protein_transformer = protein_transformer(nhead, dropout, dim, hid_size, layers_trans, max_len=n_word)
        self.softmax = nn.Softmax(dim=1)
        self.ELU = nn.ELU(1.0)

    def fingerprint_gcn(self, smileadjacency, fingerprints, dropout):
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        substrate_vectors = self.gcn(fingerprint_vectors, smileadjacency)
        return substrate_vectors

    def seq_transformer(self, words, dropout):
        words = words.unsqueeze(0)
        words = self.embed_wordTrans(words)
        seq_vectors = self.protein_transformer(words)
        return seq_vectors

    def protein_gcn(self, adjacency, words, dropout):
        adjacency = torch.tensor(adjacency.toarray()).cuda()
        word_vectors = self.embed_wordGCN(words)
        protein_vectors = self.gcn(word_vectors, adjacency)
        return protein_vectors

    def forward(self, inputs, layer_output, dropout):
        fingerprints, smileadjacency, words, seqadjacency = inputs

        substrate_vectors = self.fingerprint_gcn(smileadjacency, fingerprints, dropout)
        substrate_vectors = torch.unsqueeze(torch.mean(substrate_vectors, 0), 0)

        seq_vectors = self.seq_transformer(words, dropout)
        seq_vectors = torch.unsqueeze(torch.mean(seq_vectors, 0), 0)

        protein_vectors = self.protein_gcn(seqadjacency, words, dropout)
        protein_vectors = torch.unsqueeze(torch.mean(protein_vectors, 0), 0)

        cat_vector = torch.cat((substrate_vectors, protein_vectors, seq_vectors), 1)

        for j in range(layer_output):
            cat_vector = F.relu(cat_vector)
            cat_vector = F.dropout(cat_vector, dropout, training=self.training)
            cat_vector = self.W_out[j](cat_vector)

        cat_vector = F.relu(cat_vector)
        cat_vector = F.dropout(cat_vector, dropout, training=self.training)
        interaction = self.W_interaction(cat_vector)
        interaction = torch.squeeze(interaction, 0)
        return interaction
