from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, attention, tradeoff_context_embed):
        print("vocab_size", vocab_size)
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = 2
        self.tradeoff = tradeoff_context_embed
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.dropout = 0.5
        self.attention = attention(self.hidden_size, self.n_layers).to(device)
        if self.tradeoff is not None:
            self.context_shrink = nn.Linear(self.hidden_size, self.embed_size * self.tradeoff).to(device)
            self.gru = nn.GRU(self.embed_size * self.tradeoff + self.embed_size, self.hidden_size, self.n_layers, dropout=self.dropout).to(device)
        else:
            self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout).to(device)
        self.out = nn.Linear(self.hidden_size, vocab_size).to(device)

    def forward(self, in_char, hidden, encoder_output, src_len, prev_attn):
        width = encoder_output.shape[0]
        enc_len = src_len.cpu().numpy() * (width / src_len.cpu().numpy()[0])
        enc_len = enc_len + 0.999
        enc_len = enc_len.astype('int')
        attn_weights = self.attention(hidden, encoder_output, enc_len, prev_attn)

        encoder_output_b = encoder_output.permute(1, 2, 0).to(device)
        context = torch.bmm(encoder_output_b, attn_weights)
        context = context.squeeze(2)
        if self.tradeoff is not None:
            context = self.context_shrink(context)

        top1 = in_char.topk(1)[1]  # batch, 1
        embed_char = self.embedding(top1)  # batch,1,embed
        embed_char = embed_char.squeeze(1).to(device)

        in_dec = torch.cat((embed_char, context), 1)
        in_dec = in_dec.unsqueeze(0)
        output, latest_hidden = self.gru(in_dec, hidden)
        output = output.squeeze(0)
        output = self.out(output)
        return output, latest_hidden, attn_weights.squeeze(2)
