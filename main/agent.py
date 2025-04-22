import torch


class Actor(torch.nn.Module):

    def __init__(self, action_space, feature_space):
        super().__init__()
        self.in_layer = torch.nn.Linear(feature_space, 64)
        self.mid_layer = torch.nn.Linear(64, 64)
        self.out_layer = torch.nn.Linear(64, action_space)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):   
        x = torch.log(x + 1)
        a1 = self.in_layer(x)
        z1 = self.relu(a1)
        a2 = self.mid_layer(z1)
        z2 = self.relu(a2)
        a3 = self.out_layer(z2)
        out = self.softmax(a3)
        return out


class Critic(torch.nn.Module):

    def __init__(self, feature_space):
        super().__init__()
        self.in_layer = torch.nn.Linear(feature_space, 64)
        self.mid_layer = torch.nn.Linear(64, 64)
        self.out_layer = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        x = torch.log(x + 1)
        a1 = self.in_layer(x)
        z1 = self.relu(a1)
        a2 = self.mid_layer(z1)
        z2 = self.relu(a2)
        a3 = self.out_layer(z2)
        out = a3
        return out