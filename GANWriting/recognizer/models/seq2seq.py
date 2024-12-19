import torch
from torch import nn
import random

##print_shape_flag = False

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, tar, src_len, teacher_rate, train=True):
        tar = tar.permute(1, 0)  # time_s, batch
        batch_size = src.size(0)
        #print(f"src shape: {src.shape}, tar shape: {tar.shape}, src_len shape: {src_len.shape}")

        outputs = torch.zeros(1, batch_size, self.vocab_size, device=device, requires_grad=True)  # Adjusted for a single output

        out_enc, hidden_enc = self.encoder(src, src_len)
        #print(f"out_enc shape: {out_enc.shape}")
        #print(f"hidden_enc shape: {hidden_enc.shape}")

        output = self.one_hot(tar[0].detach())
        hidden = hidden_enc
        attn_weights = torch.zeros(out_enc.shape[1], out_enc.shape[0], requires_grad=True).to(device)
        #print(f"Initial output shape: {output.shape}, hidden shape: {hidden.shape}, attn_weights shape: {attn_weights.shape}")

        output, hidden, attn_weights = self.decoder(output, hidden, out_enc, src_len, attn_weights)
        #print(f"Decoded output shape: {output.shape}, hidden shape: {hidden.shape}, attn_weights shape: {attn_weights.shape}")

        new_outputs = torch.zeros_like(outputs)
        new_outputs[0] = output
        outputs = new_outputs
        #print(f"outputs shape: {outputs.shape}")

        return outputs, attn_weights

    def one_hot(self, src):
        ones = torch.eye(self.vocab_size).to(device)
        #print(f"src (indices) shape: {src.shape}, src: {src}, vocab_size: {self.vocab_size}")
        
        # Check for any out-of-bounds indices
        if src.max() >= self.vocab_size or src.min() < 0:
            raise ValueError(f"src contains out-of-bounds indices: {src}")
        
        return ones.index_select(0, src)






