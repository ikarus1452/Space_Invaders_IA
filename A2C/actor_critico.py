import torch
import torch.nn as nn
import torch.autograd as autograd 
from torch.distributions import Categorical


class ActorCnn(nn.Module):
    """
    Red neuronal convolucional para el actor en el algoritmo Actor-Critic.

    Atributos:
        input_shape (tuple): Dimensiones del estado de entrada
        num_actions (int): Número de posibles acciones.
        check (str): Ruta para guardar el modelo.
        features (nn.Sequential): Capas convolucionales para procesar la entrada.
        fc (nn.Sequential): Capas totalmente conectadas para la salida de la acción.
    """

    def __init__(self, input_shape, num_actions):
        """
        Inicializa la red neuronal del actor.

        Parámetros:
            input_shape (tuple): Dimensiones del estado de entrada
            num_actions (int): Número de posibles acciones
        """
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.check = 'Actor.pt' 
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Aplicar Dropout
            nn.Linear(512, self.num_actions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        Propagación hacia adelante para generar la acción.

        Parámetros:
            x (Tensor): El tensor de entrada.

        Retorna:
            Categorical: Distribución de probabilidad sobre las posibles acciones.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        dist = Categorical(x) ## Convierte la salida en una distribución de probabilidad sobre las posibles acciones usando Categorical.
        return dist
    
    def feature_size(self):
        """
        Calcula el tamaño de la salida de las capas convolucionales.

        Retorna:
            int: Tamaño de la salida de las capas convolucionales.
        """
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def save(self):
        """
        Guarda el estado de la red neuronal en un archivo.
        """

        torch.save(self.state_dict(), self.check)
    
    def load(self):
        """
        Carga un estado guardado de la red neuronal desde un archivo.
        """
        self.load_state_dict(torch.load(self.check))
    

class CriticCnn(nn.Module):
    """
    Red neuronal convolucional para el crítico en el algoritmo Actor-Critic.
    Similar a la clase ActorCnn, pero con una salida diferente para estimar el valor del estado.
    """
    def __init__(self, input_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape
        self.check = 'Critic.pt'
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Aplicar Dropout
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def save(self):
        torch.save(self.state_dict(), self.check)
    
    def load(self):
        self.load_state_dict(torch.load(self.check))