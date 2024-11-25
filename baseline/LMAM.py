import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=4, score_function='scaled_dot_product', dropout=0.6):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class MatchingAttention(nn.Module):

    def __init__(self, rank, cand_dim, alpha_dim=None, att_type='general2'):
        super(MatchingAttention, self).__init__()
        assert rank == cand_dim
        self.rank = rank
        self.cand_dim = cand_dim
        self.transform = nn.Linear(cand_dim, rank, bias=True)
        torch.nn.init.normal_(self.transform.weight, std=0.01)


    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, rank)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        M_ = M.permute(1, 2, 0)  # batch, rank, seqlen
        x_ = self.transform(x).unsqueeze(1)  # batch, 1, rank
        mask_ = mask.unsqueeze(2).repeat(1, 1, self.rank).transpose(1, 2)  # batch, seq_len, rank
        M_ = M_ * mask_
        alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
        alpha_ = torch.tanh(alpha_)
        alpha_ = F.softmax(alpha_, dim=2)
        alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
        alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
        # import ipdb;ipdb.set_trace()

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha

def Matching(matchatt, emotions, modals, umask):
    att_emotions = []
    seq_length = emotions.size(0)
    for modal in modals:
        for t in modal:
            att_em, alpha_ = matchatt(emotions, t, mask=umask)
            att_emotions.append(att_em.unsqueeze(0))
    att_emotions = torch.cat(att_emotions, dim=0)
    hidden = att_emotions[:seq_length] + att_emotions[seq_length: 2 * seq_length] + att_emotions[2 * seq_length: 3 * seq_length]
    hidden = F.gelu(hidden) + emotions
    return hidden