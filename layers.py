"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from util import masked_sigmoid
from util import discretize


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.cnn = nn.Conv1d(char_vectors.size(
            1), hidden_size, kernel_size=5, bias=True)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, 2 * hidden_size)

    def forward(self, w, c):
        batch_size, sent_len, word_len = c.size()

        c = self.char_embed(c.view(-1, word_len))
        c = F.dropout(c, self.drop_prob, self.training)  # apply dropout

        c_emb = self.cnn(c.permute(0, 2, 1))  # Conv1D Layer
        c_emb = torch.max(F.relu(c_emb), dim=-1)[0]  # Maxpool
        c_emb = c_emb.view(batch_size, sent_len, self.hidden_size)

        w_emb = self.word_embed(w)   # (batch_size, seq_len, embed_size)
        w_emb = F.dropout(w_emb, self.drop_prob, self.training)
        w_emb = self.proj(w_emb)  # (batch_size, seq_len, hidden_size)

        # concatenate word and char embeddings
        emb = torch.cat((c_emb, w_emb), dim=-1)

        emb = self.hwy(emb)   # (batch_size, seq_len, 2 * hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        # (batch_size, c_len, q_len)
        s = self.get_similarity_matrix(c, q)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        # (batch_size, c_len, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)
        # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        # (bs, c_len, hid_size)
        c = F.dropout(c, self.drop_prob, self.training)
        # (bs, q_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class ConvBlock(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvBlock, self).__init__()

        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        out = self.depthwise(x)
        out = self.pointwise(out)

        return torch.transpose(F.relu(out + x), 1, 2)


class FFNBlock(nn.Module):

    def __init__(self, d_model, dropout=0.1, hidden_size=8):
        super(FFNBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn_layer = nn.Linear(d_model, d_model, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        norm_out = self.norm(x)
        ffn_layer_out = self.ffn_layer(norm_out)

        return self.dropout_layer(F.relu(x + ffn_layer_out))


class SelfAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfAttentionBlock,  self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attn_layer = nn.MultiheadAttention(
            d_model, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm_out = self.norm(x)

        attn_output, attn_output_weights = self.self_attn_layer(
            norm_out, norm_out, norm_out)

        return self.dropout(x + attn_output)


class PositionalEncoding(nn.Module):
    """ Position Encoder which injects positional structure and information of to the input sequence.
    This particular implementation was derived from the one implemented on the pytorch transformer in the pytorch documentation(https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model)
    Args:
        d_model () :
        dropout () :
        max_len () :
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddingResizer(nn.Module):
    """ Resizes input embedding to hidden size of 128 that can be passed to the convolution blocks
    Args:
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1, bias=False):
        super(EmbeddingResizer, self).__init__()

        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return torch.transpose(self.out(x), 1, 2)


class StackedEncoder(nn.Module):
    """ Base module for the Embedding and Model Encoder used in QANet.
    Args:
    """

    def __init__(self, num_conv_blocks, kernel_size, num_heads=8, d_model=128, dropout=0.1):

        super(StackedEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.conv_blocks = nn.ModuleList([ConvBlock(d_model, d_model, kernel_size)
                                          for _ in range(num_conv_blocks)])

        self.self_attn_block = SelfAttentionBlock(d_model, num_heads, dropout)
        self.ffn_block = FFNBlock(d_model)

        self.num_conv_blocks = num_conv_blocks

        self.dropout = dropout

    def forward(self, x):
        x = self.pos_encoder(x)

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.self_attn_block(x)

        return self.ffn_block(x)


class FV(nn.Module):
    """Front Verification layer utilized as part of Retrospective reader
    to augment our QANet by addressing the question of answerability
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
    """

    def __init__(self, hidden_size):
        super(FV, self).__init__()

        self.verify_linear = nn.Linear(hidden_size * 3, 1)

    def forward(self, M_1, M_2, M_3, mask):
        #linear layer
        M_X = self.verify_linear(torch.cat((M_1, M_2, M_3), dim=-1))
        #produce logits
        sq1 = masked_sigmoid(torch.squeeze(M_X), mask, log_sigmoid=False)
    
        y_i = torch.squeeze(sq1[:,0])

        return y_i


class IntensiveOutput(nn.Module):
    """Outputs the results of running the sample through the intensive module, implementing internal front verification and a span predicition
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
    """

    def __init__(self, hidden_size):
        super(IntensiveOutput, self).__init__()
        self.ifv = FV(hidden_size)
        # need to make these the size of M_i
        self.Ws = nn.Parameter(torch.zeros(hidden_size * 2, 1))
        self.We = nn.Parameter(torch.zeros(hidden_size * 2, 1))

        self.softmax = nn.Softmax(0)

    def forward(self, M_1, M_2, M_3, mask):
        y_i = self.ifv(M_1, M_2, M_3, mask)
        logits_1 = torch.squeeze(torch.cat((M_1, M_2), dim=-1) @ self.Ws)
        logits_2 = torch.squeeze(torch.cat((M_1, M_3), dim=-1) @ self.We)

        log_p1 = masked_softmax(logits_1, mask, dim=-1, log_softmax=True)
        log_p2 = masked_softmax(logits_2, mask, dim=-1, log_softmax=True)
        return y_i, log_p1, log_p2


class SketchyOutput(nn.Module):
    """Outputs the results of running the sample throuhg the sketchy reading module, implements external front verification
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
    """

    def __init__(self, hidden_size):
        super(SketchyOutput, self).__init__()
        self.efv = FV(hidden_size)

    def forward(self, M_1, M_2, M_3, mask):
        y_i = self.efv(M_1, M_2, M_3, mask)

        return y_i


class RV_TAV(nn.Module):
    """Rear Verification and Threshold Answer Verification layer utilized as part of Retrospective reader
    to augment our QANet by combining the answerability determined by our sketchy model and ur intensive
    model either returning a span or no answer at all.
    """

    def __init__(self):
        super(RV_TAV, self).__init__()

        # Allows us to train weights for RV
        self.beta = nn.Parameter(torch.zeros(1) + 0.5)
        # Allows us to train Threshold for TAV
        self.ans = nn.Parameter(torch.zeros(1) + 0.75)

    def forward(self, sketchy_prediction, intensive_prediction, log_p1, log_p2, max_len=15, use_squad_v2=True):
        starts, ends = discretize(
            log_p1.exp(), log_p2.exp(), max_len, use_squad_v2)
        # Combines answerability estimate from both the sketchy and intensive models
        pred_answerable = self.beta * intensive_prediction + \
            (1-self.beta) * sketchy_prediction
        # Calcultes how certain we are of intesives prediction
        has = torch.tensor([log_p1[x, starts[x]] * log_p2[x, ends[x]] for x in range(64)]).to(device='cuda')
        null = (log_p1[:, 0] * log_p2[:, 0]).to(device='cuda')
        span_answerable = null - has
        # Combines our answerability with our certainty
        answerable = pred_answerable + span_answerable 
        log_p1[answerable > self.ans] = 0.0
        log_p2[answerable > self.ans] = 0.0
        return log_p1, log_p2
        
