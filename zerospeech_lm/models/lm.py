import torch
import torch.nn as nn
import torch.functional as F


# QRNN layer with window size = 1
# based on https://github.com/salesforce/pytorch-qrnn
class ForgetMult(torch.nn.Module):
    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)
    
    
class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, output_gate=True, use_cuda=True):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda

        self.linear = nn.Linear(self.input_size, 
                                3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)
        self.forget_mult = ForgetMult()
    
    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()
        source = X

        Y = self.linear(source)
        
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
            
        Z = torch.tanh(Z).contiguous()
        F = torch.sigmoid(F).contiguous()
        C = self.forget_mult(F, Z, hidden)

        if self.output_gate:
            H = torch.sigmoid(O) * C
        else:
            H = C

        return H, C[-1:, :, :]
    

class QRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, layers=None, **kwargs):
        super(QRNN, self).__init__()

        self.layers = nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.dropout = dropout

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = F.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden


class QRNN_model(nn.Module):
    def __init__(self, vocab_size, args, dropout_rate=0.1):
        super(QRNN_model, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.fc_hid_dim = self.hidden_dim // 5
        self.num_layers = args.num_layers
        self.dropout_rate = dropout_rate
        self.bsz = args.bsz
        self.device = args.device
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout_in = nn.Dropout(self.dropout_rate)
        self.qrnn = QRNN(self.emb_dim, self.hidden_dim, self.num_layers)
        self.fc1 = nn.Linear(self.hidden_dim, self.fc_hid_dim)
        self.fc2 = nn.Linear(self.fc_hid_dim, self.vocab_size)
        self.dropout_out = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(dim=2)
            
    def forward(self, x, hidden_state):
        # hidden state in the forward needs to be (n_layers, bsz, hidden_dim)
        # while outside the moel (bsz, n_layers, hidden_dim) because of DataParallel and batchifier
        x = self.embedding(x)
        x = self.dropout_in(x)
        x, hidden_state = self.qrnn(x.permute(1,0,2).contiguous(), hidden_state.permute(1,0,2).contiguous())
        x = self.fc1(x.permute(1,0,2).contiguous())
        x = self.fc2(x)
        x = self.dropout_out(x)
        return self.softmax(x), hidden_state.permute(1,0,2).contiguous()


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, args, dropout_rate=0.1):
        super(LSTM_model, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.fc_hid_dim = self.hidden_dim // 5
        self.num_layers = args.num_layers
        self.dropout_rate = dropout_rate
        self.device = args.device
        if args.bidirectional:
            self.hidden_scale = 2
        else:
            self.hidden_scale = 1
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout_in = nn.Dropout(self.dropout_rate)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim // self.hidden_scale, self.num_layers, batch_first=True, bidirectional=args.bidirectional)
        self.fc1 = nn.Linear(self.hidden_dim, self.fc_hid_dim)
        self.fc2 = nn.Linear(self.fc_hid_dim, self.vocab_size)
        self.dropout_out = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, x, hidden_state):
        # hidden state in the forward needs to be (n_layers, bsz, hidden_dim)
        # while outside the moel (bsz, n_layers, hidden_dim) because of DataParallel and batchifier
        hidden_state = tuple([h.permute(1,0,2).contiguous() for h in hidden_state])
        x = self.embedding(x)
        self.lstm.flatten_parameters()
        x = self.dropout_in(x)
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout_out(x)
        hidden_state = tuple([h.permute(1,0,2).contiguous() for h in hidden_state])
        return self.softmax(x), hidden_state