from network.network import Network
from pykdtree.kdtree import KDTree
import torch
from torch import nn
import torch.nn.functional as F

class MLP(Network):

    def __init__(self,config):
        self.h_sizes = []
        self.out_size = 1
        super().__init__(config)
        # Hidden layers
        hidden = []
        for k in range(len(self.h_sizes)-1):
            hidden.append(nn.Linear(self.h_sizes[k], self.h_sizes[k+1]).cuda())

        self.hidden = torch.nn.ModuleList(hidden)
        # Output layer
        self.out = nn.Linear(self.h_sizes[-1], self.out_size).cuda()


    def forward(self, x):
        x = x["coords"]
        # Feedforward
        for layer in self.hidden:
            x = F.softplus(layer(x), beta=100)

        output = self.out(x)
        output = torch.sigmoid(output)

        return {"sdf":output}
