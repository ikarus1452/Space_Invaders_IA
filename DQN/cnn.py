import torch as T
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import os

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions, checkpoint, name,arq):
        super(DQNCnn, self).__init__()
        self.arq = arq
        self.check = 'policy_DQN.pt'
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        
        # Capas convolucionales
        self.features1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Capas fully connected
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

        
        
    def forward(self, x):
        if self.arq == 1:
            x = self.features1(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            
        elif self.arq ==2:
            x = self.features2(x)
            x = x.view(x.size(0), -1)
            x = self.fc2(x)
            
        elif self.arq ==3:
            x = self.features1(x)
            x = x.view(x.size(0), -1)
            x = self.fc2(x)
        return x
    
    def feature_size(self):
        if self.arq == 1:
            x = self.features1(autograd.Variable(T.zeros(1, *self.input_shape))).view(1, -1).size(1)
        elif self.arq == 2:
            x = self.features2(autograd.Variable(T.zeros(1, *self.input_shape))).view(1, -1).size(1)
        elif self.arq == 3:
            x = self.features1(autograd.Variable(T.zeros(1, *self.input_shape))).view(1, -1).size(1)

        return x
    
    
    def save(self):
        T.save(self.state_dict(), self.check)
    
    def load(self):
        self.load_state_dict(T.load(self.check))