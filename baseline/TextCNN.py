import torch.nn as nn
import torch.nn.functional as F
import torch
from .LMAM import MatchingAttention, Attention, Matching

class TextCNN(nn.Module):

    def __init__(self, num_classes, attention):
        super(TextCNN, self).__init__()
        self.attention = attention
        self.textf_input = nn.Conv1d(1024, 512, kernel_size=1, padding=0, bias=True)
        self.acouf_input = nn.Conv1d(1582, 512, kernel_size=1, padding=0, bias=True)
        self.visuf_input = nn.Conv1d(342, 512, kernel_size=1, padding=0, bias=True)
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        self.matchatt = MatchingAttention(512, 512, att_type='general2')
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.linear4= nn.Linear(1536, 256)
        self.num_classes = num_classes

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
        textf = self.textf_input(textf.permute(1, 2, 0)).permute(2, 0, 1)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).permute(2, 0, 1)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).permute(2, 0, 1)
        textf = F.gelu(self.linear1(textf))
        acouf = F.gelu(self.linear1(acouf))
        visuf = F.gelu(self.linear1(visuf))
        if self.attention == "LMAM":
            x = Matching(self.matchatt, textf, [textf, acouf, visuf], umask)
        else:
            x = torch.cat((textf, acouf, visuf), dim=-1)
        # project the features to the labels
        x = F.gelu(self.linear4(x))
        x = self.linear3(x)
        log_prob = F.log_softmax(x, 2)
        return log_prob
