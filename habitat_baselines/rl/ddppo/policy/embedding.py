import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEdgesEmbedding(nn.Module):
    def __init__(self):
        super(SpatialEdgesEmbedding, self).__init__()
        self.spatial_edges_embedding = nn.Linear(72, 1800)
    
    def forward(self, x):
        res = []
        for _x in torch.split(x, 4):
          _x = _expand_tensor2D(_x, 6, 3)
          assert _x.size() == (4, 6, 3)
          _x = _x.view(-1)
          _x = self.spatial_edges_embedding(_x)
          _x = F.relu(_x)
          _x = _x.view((180, 5, 2))
          res.append(_x)
        return torch.split(torch.cat(res), 180)


class TemporalEdgesEmbedding(nn.Module):
    def __init__(self):
        super(TemporalEdgesEmbedding, self).__init__()
        self.temporal_edges_embedding = nn.Linear(8, 360)
    
    def forward(self, x):
        res = []
        for _x in torch.split(x, 4):
          _x = _expand_tensor(_x, 2)
          assert _x.size() == (4, 2)
          _x = _x.view(-1)
          _x = self.temporal_edges_embedding(_x)
          _x = F.relu(_x)
          _x = _x.view((180, 1, 2))
          res.append(_x)
        return torch.split(torch.cat(res), 180)

class RobotNodeEmbedding(nn.Module):
    def __init__(self):
        super(RobotNodeEmbedding, self).__init__()
        self.robot_node_embedding = nn.Linear(2048, 1260)
    
    def forward(self, x):
        res = []
        for _x in torch.split(x, 4):
          _x = _expand_tensor(_x, 512)
          assert _x.size() == (4, 512)
          _x = _x.view(-1)
          _x = self.robot_node_embedding(_x)
          _x = F.relu(_x)
          _x = _x.view((180, 1, 7))
          res.append(_x)
        return torch.split(torch.cat(res), 180)

def _expand_tensor(inp, width):
  length = int(inp.data.shape[0])
  paddings = 4 - length
  return torch.cat((inp, torch.zeros((paddings, width), device=inp.device)))

def _expand_tensor2D(inp, x, y):
  length = int(inp.data.shape[0])
  paddings = 4 - length
  return torch.cat((inp, torch.zeros((paddings, x, y), device=inp.device)))