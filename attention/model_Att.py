import torch
import torch.nn as nn
import random

seed_num = 5
torch.manual_seed(seed_num)
random.seed(seed_num)

# ''' attention model '''
class BidirectionalLSTM(nn.Module):

    def __init__(self, nCh, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        # attention block
        self.rnn1 = nn.LSTM(nCh, nCh, bidirectional=False)

        self.spatial = nn.Softmax(dim=1)
        self.temporal = nn.Softmax(dim=0)

        # self.spatial = nn.Sigmoid()
        # self.temporal = nn.Sigmoid()

        self.rnn2 = nn.LSTM(nCh, nHidden, bidirectional=True)
        self.embedding1 = nn.Linear(nHidden*2, nOut)
        self.embedding2 = nn.Linear(nHidden*2, nOut)

    def forward(self, input):
        # input of shape (seq_len, batch, input_size)
        input = input.permute(2, 0, 1)
        # print("block input size = %s" % (str(input.size())))

        low, _ = self.rnn1(input)
        T, b, h = low.size()
        # print("block input size = %s, %s, %s" % (T, b, h))

        s_mask_low = torch.sum(low, dim=0)
        s_mask = self.spatial(s_mask_low)
        # print("block s_mask size = %s" % (str(s_mask_low.size())))

        t_mask_low = torch.sum(low, dim=2)
        t_mask = self.temporal(t_mask_low)
        # print("block t_mask size = %s" % (str(t_mask_low.size())))
        #
        att_input = input * s_mask
        # print("block att input size = %s" % (str(att_input.size())))
        recurrent, _ = self.rnn2(att_input)
        T, b, h = recurrent.size()
        # print("block input size = %s, %s, %s" % (T, b, h))

        att_recurrent = recurrent * t_mask.unsqueeze(-1).repeat(1,1,h)

        # print("block att recurrent size = %s" % (str(att_recurrent.size())))
        t_rec = att_recurrent.reshape(T * b, h)
        # print("block t_rec shape = %s" % (str(t_rec.shape)))

        out1 = self.embedding1(t_rec)  # [T * b, nOut]
        out1 = out1.view(T, b, -1)
        # print("block output size = %s" % (str(out1.size())))
        out1 = out1.permute(1, 2, 0)

        out2 = self.embedding2(t_rec)  # [T * b, nOut]
        out2 = out2.view(T, b, -1)
        # print("block output size = %s" % (str(out2.size())))
        out2 = out2.permute(1, 2, 0)

        return out1, out2, t_mask, s_mask