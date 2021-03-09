import torch

from layers import ConvBlock, EmbeddingResizer, PositionalEncoding, StackedEncoder

sent = torch.randn((400, 1))

sent = sent.unsqueeze(0)

conv = ConvBlock(128, 128, 7)
resizer = EmbeddingResizer(400, 128)
# pos_econder = PositionalEncoding(128, 0.1)
encoder = StackedEncoder(7, 7)
out1 = resizer(sent)


print(out1.size())

# pos_out = pos_econder(out1)

# print(pos_out.size())

# out = conv(pos_out)
out = encoder(out1)

print(out.size())
