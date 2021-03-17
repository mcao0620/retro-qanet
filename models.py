"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        hidden_size *= 2 # update hidden size for other layers due to char embeddings

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):

    def __init__(self, word_vectors, char_vectors, hidden_size, device, drop_prob=0.):
        super(QANet, self).__init__()

        self.device = device

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        hidden_size *= 2    # update hidden size for other layers due to char embeddings

        self.c_resizer = layers.EmbeddingResizer(in_channels=hidden_size,
                                               out_channels=128)
                    
        self.q_resizer = layers.EmbeddingResizer(in_channels=hidden_size,
                                               out_channels=128)

        self.model_resizer = layers.EmbeddingResizer(in_channels=512,
                                                     out_channels=128)

        self.enc = layers.StackedEncoder(num_conv_blocks=4,
                                         kernel_size=7,
                                         dropout=drop_prob,
                                         device=self.device)     # embedding encoder layer

        self.att = layers.BiDAFAttention(hidden_size=128,
                                         drop_prob=drop_prob)     # context-query attention layer

        self.model_encoder_layers = nn.ModuleList([layers.StackedEncoder(num_conv_blocks=2,
                                                                         kernel_size=7,
                                                                         dropout=drop_prob,
                                                                         device=self.device) for _ in range(7)])

        self.out = layers.QANetOutput(hidden_size=128)     # output layer

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # c_mask_3d = torch.eq(cw_idxs, 1).float()
        # q_mask_3d = torch.eq(qw_idxs, 1).float()

        # (batch_size, c_len, hidden_size)
        c_emb = self.emb(cw_idxs, cc_idxs)
        # (batch_size, q_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)

        c_emb = self.c_resizer(c_emb)
        q_emb = self.q_resizer(q_emb)

        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        att = self.model_resizer(att)

        mod1 = att

        for layer in self.model_encoder_layers:
            mod1 = layer(mod1, c_mask)

        mod2 = mod1

        for layer in self.model_encoder_layers:
            mod2 = layer(mod2, c_mask)

        mod3 = mod2

        for layer in self.model_encoder_layers:
            mod3 = layer(mod3, c_mask)

        out = self.out(mod1.transpose(1, 2), mod2.transpose(1, 2), mod3.transpose(1, 2), c_mask)

        return out