"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import util


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

    def __init__(self, word_vectors, char_vectors, hidden_size, num_heads=8, drop_prob=0.):
        super(BiDAF, self).__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        hidden_size *= 2  # update hidden size for other layers due to char embeddings

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

        # (batch_size, c_len, hidden_size)
        c_emb = self.emb(cw_idxs, cc_idxs)
        # (batch_size, q_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class SketchyReader(nn.Module):

    def __init__(self, word_vectors, char_vectors, hidden_size, num_heads, char_embed_drop_prob, drop_prob=0.1):
        super(SketchyReader, self).__init__()
        '''class QANet(nn.Module):

        def __init__(self, word_vectors, char_vectors, hidden_size, device, drop_prob=0.):
        super(QANet, self).__init__()

        self.device = device'''

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    char_embed_drop_prob=char_embed_drop_prob,
                                    word_embed_drop_prob=drop_prob)

        hidden_size *= 2    # update hidden size for other layers due to char embeddings

        self.c_resizer = layers.Initialized_Conv1d(hidden_size, 128)

        self.q_resizer = layers.Initialized_Conv1d(hidden_size, 128)

        self.model_resizer = layers.Initialized_Conv1d(512, 128)

        self.enc = layers.StackedEncoder(num_conv_blocks=4,
                                         kernel_size=7,
                                         num_heads=num_heads,
                                         dropout=drop_prob)     # embedding encoder layer
        self.att = layers.BiDAFAttention(hidden_size=128,
                                         drop_prob=drop_prob)     # context-query attention layer

        # self.mod1 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer

        # self.mod2 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer

        # self.mod3 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer
        self.model_encoder_layers = nn.ModuleList([layers.StackedEncoder(num_conv_blocks=2,
                                                                         kernel_size=7,
                                                                         dropout=drop_prob) for _ in range(7)])

        self.out = layers.SketchyOutput(hidden_size=128)     # output layer

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

        c_emb = self.c_resizer(c_emb.transpose(1, 2))
        q_emb = self.q_resizer(q_emb.transpose(1, 2))

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_mask, 1, 1)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_mask, 1, 1)

        att = self.att(c_enc.transpose(1, 2), q_enc.transpose(1, 2),
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        att = att.transpose(1, 2)
        att = self.model_resizer(att)

        mod1 = att

        for i, layer in enumerate(self.model_encoder_layers):
            mod1 = layer(mod1, c_mask, i*(2+2)+1, 7)

        mod2 = mod1

        for i, layer in enumerate(self.model_encoder_layers):
            mod2 = layer(mod2, c_mask, i*(2+2)+1, 7)

        mod3 = mod2

        for i, layer in enumerate(self.model_encoder_layers):
            mod3 = layer(mod3, c_mask, i*(2+2)+1, 7)

        # mod1 = self.mod1(att)        # (batch_size, c_len, 2 * hidden_size)
        # mod2 = self.mod2(mod1)        # (batch_size, c_len, 2 * hidden_size)
        # mod3 = self.mod3(mod2)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(mod1, mod2, mod3, c_mask)

        return out


class IntensiveReader(nn.Module):

    def __init__(self, word_vectors, char_vectors, hidden_size, num_heads, char_embed_drop_prob, drop_prob=0.):
        super(IntensiveReader, self).__init__()
        '''class QANet(nn.Module):

        def __init__(self, word_vectors, char_vectors, hidden_size, device, drop_prob=0.):
        super(QANet, self).__init__()

        self.device = device'''

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    char_embed_drop_prob=char_embed_drop_prob,
                                    word_embed_drop_prob=drop_prob)

        hidden_size *= 2    # update hidden size for other layers due to char embeddings

        self.c_resizer = layers.Initialized_Conv1d(hidden_size, 128)

        self.q_resizer = layers.Initialized_Conv1d(hidden_size, 128)

        self.model_resizer = layers.Initialized_Conv1d(512, 128)

        self.enc = layers.StackedEncoder(num_conv_blocks=4,
                                         kernel_size=7,
                                         num_heads=num_heads,
                                         dropout=drop_prob)     # embedding encoder layer
        self.att = layers.BiDAFAttention(hidden_size=128,
                                         drop_prob=drop_prob)     # context-query attention layer

        # self.mod1 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer

        # self.mod2 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer

        # self.mod3 = layers.StackedEncoder(num_conv_blocks=2,
        #                                  kernel_size=7,
        #                                  dropout=drop_prob)     # model layer
        self.model_encoder_layers = nn.ModuleList([layers.StackedEncoder(num_conv_blocks=2,
                                                                         kernel_size=7,
                                                                         dropout=drop_prob) for _ in range(7)])

        self.out = layers.IntensiveOutput(hidden_size=128)     # output layer

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

        c_emb = self.c_resizer(c_emb.transpose(1, 2))
        q_emb = self.q_resizer(q_emb.transpose(1, 2))

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_mask, 1, 1)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_mask, 1, 1)

        att = self.att(c_enc.transpose(1, 2), q_enc.transpose(1, 2),
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        att = att.transpose(1, 2)
        att = self.model_resizer(att)

        mod1 = att

        for i, layer in enumerate(self.model_encoder_layers):
            mod1 = layer(mod1, c_mask, i*(2+2)+1, 7)

        mod2 = mod1

        for i, layer in enumerate(self.model_encoder_layers):
            mod2 = layer(mod2, c_mask, i*(2+2)+1, 7)

        mod3 = mod2

        for i, layer in enumerate(self.model_encoder_layers):
            mod3 = layer(mod3, c_mask, i*(2+2)+1, 7)

        # mod1 = self.mod1(att)        # (batch_size, c_len, 2 * hidden_size)
        # mod2 = self.mod2(mod1)        # (batch_size, c_len, 2 * hidden_size)
        # mod3 = self.mod3(mod2)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(mod1, mod2, mod3, c_mask)

        return out


class RetroQANet(nn.Module):

    """Retro-Reader over QANet

    """

    def __init__(self, word_vectors, char_vectors, hidden_size, intensive_path, num_heads, sketchy_path, gpu_ids, char_embed_drop_prob, drop_prob=0.):
        super(RetroQANet, self).__init__()

        self.sketchy = SketchyReader(word_vectors=word_vectors,
                                     char_vectors=char_vectors,
                                     hidden_size=hidden_size,
                                     num_heads=num_heads,
                                     char_embed_drop_prob=char_embed_drop_prob,
                                     drop_prob=drop_prob)
        self.sketchy = nn.DataParallel(self.sketchy, gpu_ids)
        self.sketchy, _ = util.load_model(self.sketchy, sketchy_path, gpu_ids)

        self.intensive = IntensiveReader(word_vectors=word_vectors,
                                         char_vectors=char_vectors,
                                         num_heads=num_heads,
                                         char_embed_drop_prob=char_embed_drop_prob,
                                         hidden_size=hidden_size,
                                         drop_prob=drop_prob)
        self.intensive = nn.DataParallel(self.intensive, gpu_ids)
        self.intensive, _ = util.load_model(
            self.intensive, intensive_path, gpu_ids)

        self.RV_TAV = layers.RV_TAV()

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        self.sketchy.eval()
        self.intensive.eval()

        yi_s = self.sketchy(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
        yi_i, log_p1, log_p2 = self.intensive(
            cw_idxs, qw_idxs, cc_idxs, qc_idxs)
        out = self.RV_TAV(yi_s.to(device='cuda:0'), yi_i.to(
            device='cuda:0'), log_p1.to(device='cuda:0'), log_p2.to(device='cuda:0'))

        return out
