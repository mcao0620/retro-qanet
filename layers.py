"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.cnn = nn.Conv1d(char_vectors.size(1), hidden_size, kernel_size=5, bias=True)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, 2 * hidden_size)

    def forward(self, w, c):
        batch_size, sent_len, word_len = c.size()

        c = self.char_embed(c.view(-1, word_len))
        c = F.dropout(c, self.drop_prob, self.training) # apply dropout

        c_emb = self.cnn(c.permute(0, 2, 1)) # Conv1D Layer
        c_emb = torch.max(F.relu(c_emb), dim=-1)[0] # Maxpool
        c_emb = c_emb.view(batch_size, sent_len, self.hidden_size) 

        w_emb = self.word_embed(w)   # (batch_size, seq_len, embed_size)
        w_emb = F.dropout(w_emb, self.drop_prob, self.training)
        w_emb = self.proj(w_emb)  # (batch_size, seq_len, hidden_size)
 
        emb = torch.cat((c_emb, w_emb), dim=-1) # concatenate word and char embeddings

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

        # nn.init.kaiming_normal_(self.depthwise.weight)
        # nn.init.constant_(self.depthwise.bias, 0.0)
        # nn.init.kaiming_normal_(self.pointwise.weight)
        # nn.init.constant_(self.pointwise.bias, 0.0)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        out = self.depthwise(x)
        out = self.pointwise(out)

        return torch.transpose(F.relu(out), 1, 2)


# class FFNBlock(nn.Module):

#     def __init__(self, d_model, dropout=0.1, hidden_size=8):
#         super(FFNBlock, self).__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.ffn_layer = nn.Linear(d_model, d_model, bias=True)
#         self.dropout_layer = nn.Dropout(dropout)

#     def forward(self, x):
#         norm_out = self.norm(x)
#         ffn_layer_out = self.ffn_layer(norm_out)

#         return self.dropout_layer(F.relu(x) + ffn_layer_out)

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))



class SelfAttentionBlock(nn.Module):

     def __init__(self, d_model, num_heads, dropout=0.1):
         super(SelfAttentionBlock,  self).__init__()
         self.norm = nn.LayerNorm(d_model)
         self.self_attn_layer = nn.MultiheadAttention(
             d_model, num_heads, dropout)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         norm_out = self.norm(x)
         norm_out = norm_out.permute(1,0,2)
         attn_output, attn_output_weights = self.self_attn_layer(
             norm_out, norm_out, norm_out, key_padding_mask=~mask)

         return self.dropout(x + attn_output.permute(1,0,2))

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

# class PositionalEncoding(nn.Module):
    
#     def __init__(self, model_dim, dropout, device, max_length=400):
        
#         super(PositionalEncoding, self).__init__()
        
#         self.device = device
        
#         self.model_dim = model_dim
        
#         pos_encoding = torch.zeros(max_length, model_dim)
        
#         for pos in range(max_length):
            
#             for i in range(0, model_dim, 2):
                
#                 pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2*i)/model_dim)))
#                 pos_encoding[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/model_dim)))
            
        
#         pos_encoding = pos_encoding.unsqueeze(0).to(device)
#         self.register_buffer('pos_encoding', pos_encoding)
        
    
#     def forward(self, x):
#         #print("PE shape: ", self.pos_encoding.shape)
#         #print("PE input: ", x.shape)
#         x = x + torch.autograd.Variable(self.pos_encoding[:, :x.shape[1]], requires_grad=False)
#         #print("PE output: ", x.shape)
#         return x


# class PositionalEncoding(nn.Module):
#     """ Position Encoder which injects positional structure and information of to the input sequence.
#     This particular implementation was derived from the one implemented on the pytorch transformer in the pytorch documentation(https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model)
#     Args:
#         d_model () :
#         dropout () :
#         max_len () :
#     """

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(
#             0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


class EmbeddingResizer(nn.Module):
    """ Resizes input embedding to hidden size of 128 that can be passed to the convolution blocks
    Args:
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1, bias=False):
        super(EmbeddingResizer, self).__init__()

        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size, stride=stride,
            padding=kernel_size//2, groups=in_channels, bias=bias)
        #nn.init.xavier_uniform_(self.depthwise.weight)

        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            1, stride=stride,
            padding=padding, groups=groups, bias=True)
        #nn.init.xavier_uniform_(self.pointwise.weight)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return torch.transpose(self.pointwise(self.depthwise(x)), 1, 2)

'''class MultiheadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, num_heads, device):
        
        super().__init__()
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        
        self.head_dim = self.hid_dim // self.num_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
        
    def forward(self, x, mask):
        # x = [bs, len_x, hid_dim]
        # mask = [bs, len_x]
        
        batch_size = x.shape[0]
        
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        # Q = K = V = [bs, len_x, hid_dim]
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        # [bs, len_x, num_heads, head_dim ]  => [bs, num_heads, len_x, head_dim]
        
        K = K.permute(0,1,3,2)
        # [bs, num_heads, head_dim, len_x]
        
        energy = torch.matmul(Q, K) / self.scale
        # (bs, num_heads){[len_x, head_dim] * [head_dim, len_x]} => [bs, num_heads, len_x, len_x]
        
        mask = mask.unsqueeze(1).unsqueeze(2)
        # [bs, 1, 1, len_x]
        
        #print("Mask: ", mask)
        #print("Energy: ", energy)
        
        energy = energy.masked_fill(mask == 1, -1e10)
        
        #print("energy after masking: ", energy)
        
        alpha = torch.softmax(energy, dim=-1)
        #  [bs, num_heads, len_x, len_x]
        
        #print("energy after smax: ", alpha)
        alpha = F.dropout(alpha, p=0.1)
        
        a = torch.matmul(alpha, V)
        # [bs, num_heads, len_x, head_dim]
        
        a = a.permute(0,2,1,3)
        # [bs, len_x, num_heads, hid_dim]
        
        a = a.contiguous().view(batch_size, -1, self.hid_dim)
        # [bs, len_x, hid_dim]
        
        a = self.fc_o(a)
        # [bs, len_x, hid_dim]
        
        #print("Multihead output: ", a.shape)
        return a'''
class StackedEncoder(nn.Module):
    """ Base module for the Embedding and Model Encoder used in QANet.
    Args:
    """

    def __init__(self, num_conv_blocks, kernel_size, num_heads=4, d_model=128, dropout=0.1, device="cuda:0"):

        super(StackedEncoder, self).__init__()
        #self.pos_encoder = PositionalEncoding(d_model, dropout, device)
        #self.pos_norm = nn.LayerNorm(d_model)

        self.conv_blocks = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, kernel_size)
                                          for _ in range(num_conv_blocks)])
        self.conv_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv_blocks)])

        self.self_attn_block =  nn.MultiheadAttention(d_model, num_heads, dropout)
        #self.ffn_block = FFNBlock(d_model)

        self.ffn_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.ffn_1_norm = nn.LayerNorm(d_model)
        self.ffn_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.ffn_2_norm = nn.LayerNorm(d_model)
        '''self.conv_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv_blocks)])

       # self.self_attn_block = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.self_attn_block = MultiheadAttentionLayer(d_model, num_heads, device)
        self.ffn_block = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)'''

        self.num_conv_blocks = num_conv_blocks

        self.dropout = dropout
    
    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual

    def forward(self, x, mask, l, blks):
        '''x = self.pos_encoder(x)

        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)

            if (i+1) % 2 == 0:
                x = F.dropout(x, p=self.dropout)

        x = self.self_attn_block(x, mask)

        return self.ffn_block(x)'''

        x = PosEncoder(x)
       #res = x
        #x = self.pos_norm(x)
        total_layers = (self.num_conv_blocks + 1) * blks
        for i, conv_block in enumerate(self.conv_blocks):
            res = x
            x = self.conv_norm[i](x.transpose(1, 2)).transpose(1, 2)

            if i % 2 == 0:
                x = F.dropout(x, p=self.dropout)
            x = conv_block(x)
            x = self.layer_dropout(x, res, self.dropout * float(l) / total_layers)
            l += 1
        res = x

        x = self.ffn_1_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout)
        x = x.transpose(1, 2)
        x = x.permute(1,0,2)
        x, attn_output_weights = self.self_attn_block(x, x, x, key_padding_mask=~mask)
        x = x.permute(1,0,2)
        x = x.transpose(1, 2)
        x = self.layer_dropout(x, res, self.dropout * float(l) / total_layers)
        l += 1
        res = x
        
        x = self.ffn_2_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout)
        x = self.ffn_1(x)
        x = self.ffn_2(x)
        x = self.layer_dropout(x, res, self.dropout * float(l) / total_layers)

        return x


class QANetOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QANetOutput, self).__init__()

        self.W1 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.W2 = nn.Linear(2 * hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)


    def forward(self, M_1, M_2, M_3, mask):
        begin = torch.cat([M_1, M_2], dim=2)
        begin = self.W1(begin)
        
        end = torch.cat([M_1, M_3], dim=2)
        end = self.W2(end)

        log_p1 = masked_softmax(begin.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(end.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

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
        # linear layer
        M_X = self.verify_linear(torch.cat((M_1, M_2, M_3), dim=1))

        sq1 = masked_sigmoid(torch.squeeze(M_X), mask, log_sigmoid=False)

        #answerability that takes into account answer confidence
        y_i = torch.max(sq1.T - sq1[:,0], dim=0)[0] 

        return y_i.type(torch.FloatTensor)


class IntensiveOutput(nn.Module):
    """Outputs the results of running the sample through the intensive module, implementing internal front verification and a span predicition
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
    """

    def __init__(self, hidden_size):
        super(IntensiveOutput, self).__init__()
        self.ifv = FV(hidden_size)
        # need to make these the size of M_i
        #self.Ws = nn.Linear(2 * hidden_size, 1, bias=False)
        #self.We = nn.Linear(2 * hidden_size, 1, bias=False)
        self.Ws = Initialized_Conv1d(2 * hidden_size, 1)
        self.We = Initialized_Conv1d(2 * hidden_size, 1)

        #nn.init.xavier_uniform_(self.Ws.weight)
        #nn.init.xavier_uniform_(self.We.weight)

    def forward(self, M_1, M_2, M_3, mask):

       # y_i = self.ifv(M_1, M_2, M_3, mask)
        y_i = None
    
        logits_1 = self.Ws(torch.cat((M_1, M_2), dim=1)).squeeze()
        logits_2 = self.We(torch.cat((M_1, M_3), dim=1)).squeeze()

        log_p1 = masked_softmax(logits_1, mask, dim=1, log_softmax=True)
        log_p2 = masked_softmax(logits_2, mask, dim=1, log_softmax=True)

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
        self.beta = nn.Parameter(torch.tensor[0.1])
        # Allows us to train Threshold for TAV
        self.ans = nn.Parameter(torch.tensor([0.5]))
        self.lam = nn.Parameter(torch.tensor([0.5]))

    def forward(self, sketchy_prediction, intensive_prediction, log_p1, log_p2, max_len=15, use_squad_v2=True):
        s_in = log_p1.exp()
        e_in = log_p2.exp()
        starts, ends = discretize(
            s_in, e_in, max_len, use_squad_v2)
        # Combines answerability estimate from both the sketchy and intensive models
        pred_answerable = self.beta * intensive_prediction + \
            (1-self.beta) * sketchy_prediction
        # Calcultes how certain we are of intesives prediction
        has = torch.tensor([log_p1[x, starts[x]] * log_p2[x, ends[x]]
                            for x in range(log_p1.shape[0])]).to(device='cuda')
        null = (log_p1[:, 0] * log_p2[:, 0]).to(device='cuda')
        span_answerable = null - has
        # Combines our answerability with our certainty
        not_answerable = self.lam * pred_answerable + \
            (1 - self.lam) * span_answerable
        l_p1 = log_p1.clone()
        l_p2 = log_p2.clone()
        l_p1[not_answerable > self.ans] = 0
        l_p2[not_answerable > self.ans] = 0
        return l_p1, l_p2
