import ConnectFourBitboard as g
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self):
        fc_layer_size = 256
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, fc_layer_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(fc_layer_size * 6 * 7, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, 7)        
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




