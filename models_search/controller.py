# -*- coding: utf-8 -*-
# @Date    : 2019-09-29
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_search.building_blocks_search import (CONV_TYPE, NORM_TYPE, SHORT_CUT_TYPE, SKIP_TYPE, UP_TYPE)


class Controller(nn.Module):
    def __init__(self, args, cur_stage):
        """
        init
        :param args:
        :param cur_stage: varies from 0 to ...
        """
        super(Controller, self).__init__()
        self.hid_size = args.hid_size
        self.cur_stage = cur_stage
        self.lstm = torch.nn.LSTMCell(self.hid_size, self.hid_size)
        if cur_stage:
            self.tokens = [
                len(CONV_TYPE),
                len(NORM_TYPE),
                len(UP_TYPE),
                len(SHORT_CUT_TYPE),
                len(SKIP_TYPE) ** cur_stage,
            ]
        else:
            self.tokens = [
                len(CONV_TYPE),
                len(NORM_TYPE),
                len(UP_TYPE),
                len(SHORT_CUT_TYPE),
            ]
        self.encoder = nn.Embedding(sum(self.tokens), self.hid_size)
        self.decoders = nn.ModuleList(
            [nn.Linear(self.hid_size, token) for token in self.tokens]
        )

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hid_size, requires_grad=False).cuda()

    def forward(self, x, hidden, index):
        if index == 0:
            embed = x
        else:
            embed = self.encoder(x)
        hx, cx = self.lstm(embed, hidden)

        # decode
        logit = self.decoders[index](hx)

        return logit, (hx, cx)

    def sample(self, batch_size, with_hidden=False, prev_hiddens=None, prev_archs=None):
        x = self.initHidden(batch_size)

        if prev_hiddens:
            assert prev_archs
            prev_hxs, prev_cxs = prev_hiddens
            selected_idx = np.random.choice(
                len(prev_archs), batch_size
            )  # TODO: replace=False
            selected_idx = [int(x) for x in selected_idx]

            selected_archs = []
            selected_hxs = []
            selected_cxs = []

            for s_idx in selected_idx:
                selected_archs.append(prev_archs[s_idx].unsqueeze(0))
                selected_hxs.append(prev_hxs[s_idx].unsqueeze(0))
                selected_cxs.append(prev_cxs[s_idx].unsqueeze(0))
            selected_archs = torch.cat(selected_archs, 0)
            hidden = (torch.cat(selected_hxs, 0), torch.cat(selected_cxs, 0))
        else:
            hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        entropies = []
        actions = []
        selected_log_probs = []
        for decode_idx in range(len(self.decoders)):
            logit, hidden = self.forward(x, hidden, decode_idx)
            prob = F.softmax(logit, dim=-1)  # bs * logit_dim
            log_prob = F.log_softmax(logit, dim=-1)
            entropies.append(-(log_prob * prob).sum(1, keepdim=True))  # bs * 1
            action = prob.multinomial(1)  # batch_size * 1
            actions.append(action)
            selected_log_prob = log_prob.gather(1, action.data)  # batch_size * 1
            selected_log_probs.append(selected_log_prob)

            x = action.view(batch_size) + sum(self.tokens[:decode_idx])
            x = x.requires_grad_(False)

        archs = torch.cat(actions, -1)  # batch_size * len(self.decoders)
        selected_log_probs = torch.cat(
            selected_log_probs, -1
        )  # batch_size * len(self.decoders)
        entropies = torch.cat(entropies, 0)  # bs * 1

        if prev_hiddens:
            archs = torch.cat([selected_archs, archs], -1)

        if with_hidden:
            return archs, selected_log_probs, entropies, hidden

        return archs, selected_log_probs, entropies
