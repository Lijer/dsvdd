import torch
import torch.nn as nn

# max_sent_len=35, batch_size=50, embedding_size=300
conv1 = nn.Conv1d(in_channels=320, out_channels=200, kernel_size=2)
input = torch.randn(1500, 320, 2)
# batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
# input = input.permute(0, 2, 1)
print("input:", input.size())
output = conv1(input)
print("output:", output.size())