from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from recognizer.models.vgg_tro_channel3 import vgg19_bn

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

DROP_OUT = False
LSTM = False
SUM_UP = True
PRE_TRAIN_VGG = True

class Encoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru, step, flip):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip
        self.n_layers = 2
        self.dropout = 0.5

        self.layer = vgg19_bn(PRE_TRAIN_VGG)

        if DROP_OUT:
            self.layer_dropout = nn.Dropout2d(p=0.5)
        if self.step is not None:
            self.output_proj = nn.Linear(self.height // 16 * 512 * self.step, self.height // 16 * 512)

        if LSTM:
            RNN = nn.LSTM
        else:
            RNN = nn.GRU

        if self.bi:
            self.rnn = RNN(self.height // 16 * 512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
            if SUM_UP:
                self.enc_out_merge = lambda x: x[:, :, :x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2:]
                self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
        else:
            self.rnn = RNN(self.height // 16 * 512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

    def forward(self, in_data, in_data_len, hidden=None):
        if in_data_len is None:
            raise ValueError("in_data_len cannot be None")
        
        batch_size = in_data.shape[0]
        out = self.layer(in_data)
        if DROP_OUT and self.training:
            out = self.layer_dropout(out)
        out = out.permute(3, 0, 2, 1)
        out = out.reshape(-1, batch_size, self.height // 16 * 512)
        if self.step is not None:
            time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
            out_short = torch.zeros(time_step // self.step, batch_size, n_feature * self.step, requires_grad=True).to(device)
            for i in range(0, time_step // self.step):
                part_out = [out[j] for j in range(i * self.step, (i + 1) * self.step)]
                out_short[i] = torch.cat(part_out, 1)
            out = self.output_proj(out_short)
        width = out.shape[0]
        src_len = in_data_len.cpu().numpy() * (width / self.width)
        src_len = src_len + 0.999
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.rnn(out, hidden)
        output, output_len = pad_packed_sequence(output, batch_first=False)
        if self.bi and SUM_UP:
            output = self.enc_out_merge(output)
        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_layers]
        final_hidden = hidden[hidden_idx]
        return output, final_hidden

    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2)
        matrix_out = torch.zeros(matrix_new.shape, requires_grad=True).to(device)
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1)
        return matrix_out
