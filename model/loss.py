import torch.nn.functional as F
import torch.nn as nn
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


def l2_loss(output, target):
    criterion = nn.MSELoss(size_average=True).cuda()
    return criterion(output, target)


def distance_loss(fake, gt):
    return torch.mean(torch.norm(fake - gt, dim=len(gt.shape)-1))