import torch.nn as nn
import torch.nn.functional as F
import torch
from .LMAM import MatchingAttention, Attention, Matching

class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, attention=True):

        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(input_size=2948, hidden_size=256, bidirectional=True, dropout=0.5, num_layers=4)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=256, bidirectional=True, dropout=0.5, num_layers=4)
        self.fc1 = nn.Linear(1582, 512)
        self.fc2 = nn.Linear(342, 512)

        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        if self.attention == "LMAM":
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        elif self.attention == "Attention":
            self.att = Attention(D_m)

        self.linear = nn.Linear(512, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, textf, acouf, visuf, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        textf = (r1 + r2 + r3 + r4)/4
        if self.attention == "Attention":
            emotion = torch.cat((textf, acouf, visuf), dim=-1)
            emotion, _ = self.lstm(emotion)
            hidden = F.gelu(emotion)
            hidden = F.gelu(self.linear(hidden))
        elif self.attention == "LMAM":
            textf, _ = self.lstm2(textf)
            acouf = self.fc1(acouf)
            visuf = self.fc2(visuf)
            hidden = Matching(self.matchatt, textf, [textf, acouf, visuf], umask)
            hidden = F.gelu(self.linear(hidden))
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob