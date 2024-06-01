import torch
from torch import nn

def cos_sim(self, z1, z2):
    t1 = F.normalize(z1, p=2, dim=-1, eps=1e-5)
    t2 = F.normalize(z2, p=2, dim=-1, eps=1e-5)
    sim = torch.matmul(t1, t2.t())
    return sim

def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, sim, nce):
    f = lambda x: torch.exp(x)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLossNS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz).cuda().long()
        loss = self.criterion(x, label)
        return loss

class SemiLoss(nn.Module):
    def __init__(self):
        super(SemiLoss, self).__init__()


    def forward(self, z1, z2, nce):

        f = lambda x: torch.exp(x / nce)
        refl_sim = f(torch.matmul(z1, z1.t()))
        between_sim = f(torch.matmul(z1, z2.t()))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self, z1, z2, nce):

        f = lambda x: torch.exp(x / nce)
        new_sim = f(sims)

        return -torch.log(new_sim.diag() / new_sim.sum(1))

class InfoNceLoss(nn.Module):
    def __init__(self):
        super(InfoNceLoss, self).__init__()


    def forward(self, sims, labels, nce):

        f = lambda x: torch.exp(x / nce)

        new_sim = f(sims)
        #label=torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
        return -torch.log(new_sim[labels==1] / new_sim.sum(1))

