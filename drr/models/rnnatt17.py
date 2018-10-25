# -*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/P/P17/P17-2040.pdf
# 'A Recurrent Neural Model with Attention for the Recognition of Chinese Implicit Discourse Relations'
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAtt17(nn.Module):

    def __init__(self, opts):
        # torch.nn.Embedding(m, n)
        # m represents the total number of words, and n represents the dimension of word embedding
        self.vocab_size = opts['vocab_size']

        super(RNNAtt17, self).__init__()

        self.emb = nn.Embedding(
            self.vocab_size,
            300
        )

        # single layer bidirectional lstm
        self.blstm = nn.LSTM(
            300,
            300,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.att = nn.Linear(300, 1)

        # 4 represents the number of categories
        self.proj2class = nn.Linear(300, 4)

    def forward(self, input):
        r"""

        Args
        ----------
        input : [N, 256] N groups per batch, 256 words per group
            N is batch size, 256 is the default length of arg1:arg2

            Using fixed-length sequences of 256 tokens with
            zero padding at the beginning of shorter sequences
            and truncate longer ones

        Returns
        ----------
        prob : [N, C]
            C means number of classes

        """
        batch_size = input.size(0)

        # Each word index returns a 300-dimensional vector,
        # a total of 256 word indexes for a line of text,
        # a matrix of 256x300,
        # N rows per batch, and finally a Nx256x300 tensor.
        #
        # [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
        #
        # [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。
        #
        # [[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
        embs = self.emb(input)  # [N, 256, 300]

        # Bidirectional LSTM gets Nx256x600
        # front 300 is left to right
        # rear 300 is right to left
        hids, _ = self.blstm(embs)  # [N, 256, 300 x 2]

        # left to right
        l2r_hids = hids.view(
            batch_size,
            256,
            300,
            2
        )[:, :, :, 0]

        # right to left
        r2l_hids = hids.view(
            batch_size,
            256,
            300,
            2
        )[:, :, :, 1]

        hids_sum = l2r_hids + r2l_hids  # [N, 256, 300]
        hids_sum_activated = torch.tanh(hids_sum)
        hids_proj = self.att(hids_sum_activated).squeeze(2)  # [N, 256]

        # Dim=0 by column softmax;
        # Dim=1 by line softmax
        alpha = F.softmax(hids_proj, dim=1)  # [N, 256]

        r = torch.bmm(
            hids_sum.transpose(1, 2),
            alpha.unsqueeze(2)
        ).squeeze(2)  # [N, 300]

        unnormalized_prob = self.proj2class(r)  # [N, C]
        prob = F.softmax(unnormalized_prob, dim=1)

        return prob
