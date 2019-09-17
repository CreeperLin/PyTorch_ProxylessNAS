# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss_LS(nn.Module):
    def __init__(self, eta = 0.1):
        super().__init__()
        self.eta = eta

    def forward(self, y_pred, y_true):
        n_classes = y_pred.size(1)
        # convert to one-hot
        y_true = torch.unsqueeze(y_true, 1)
        soft_y_true = torch.zeros_like(y_pred)
        soft_y_true.scatter_(1, y_true, 1)
        # label smoothing
        soft_y_true = soft_y_true * (1 - self.eta) + self.eta / n_classes * 1
        return torch.mean(torch.sum(- soft_y_true * F.log_softmax(y_pred, dim=-1), 1))

