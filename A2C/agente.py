import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class A2CAgent():
    """
    Implementa un agente Actor-Critic para el aprendizaje por refuerzo.

    Atributos:
        actor_net (Model): Modelo del actor.
        critic_net (Model): Modelo del crítico.
        log_probs, values, rewards, masks, entropies: Listas para almacenar experiencias.
        t_step (int): Contador para controlar la actualización de la red.
    """
    def __init__(self, input_shape, action_size, seed,  gamma, alpha, beta, update_every, actor_m, critic_m):
        """
        Inicializa el agente A2C con modelos específicos para el actor y el crítico.

        Parámetros:
            input_shape (tuple): Dimensiones de cada estado 
            action_size (int): Dimensión de cada acción.
            seed (int): Semilla aleatoria para reproducibilidad.
            gamma (float): Factor de descuento.
            alpha (float): Tasa de aprendizaje del actor.
            beta (float): Tasa de aprendizaje del crítico.
            update_every (int): Frecuencia de actualización de la red.
            actor_m (Model): Clase del modelo para el actor.
            critic_m (Model): Clase del modelo para el crítico.
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size) #Se crea una instancia de la red  actor
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha) 

        # Critic-Network
        self.critic_net = critic_m(input_shape)#se crea una instancia de la red critica
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = [] 
        self.values    = [] #Valores estimados por la red critica
        self.rewards   = [] # Recompensas
        self.masks     = [] #  mascaras para estados terminales
        self.entropies = [] 

        self.t_step = 0

    def step(self, state, log_prob, entropy, reward, done, next_state): 
        """
        Procesa un paso del entorno, almacenando la experiencia y aprendiendo si es necesario.

        Parámetros:
            state: El estado actual del entorno.
            log_prob: Logaritmo de la probabilidad de la acción tomada.
            entropy: Entropía de la acción tomada.
            reward: Recompensa obtenida.
            done: Indicador de finalización del episodio.
            next_state: El siguiente estado del entorno.
        """

        state = torch.from_numpy(state).unsqueeze(0)
        
        value = self.critic_net(state)
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.from_numpy(np.array([reward])))
        self.masks.append(torch.from_numpy(np.array([1 - done])))

        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
                
    def act(self, state):
        """
        Selecciona una acción para un estado dado según la política actual.

        Parámetros:
            state: El estado actual del entorno.

        Retorna:
            Una tupla de (acción, log_prob, entropía) para el estado dado.
        """
        
        state = torch.from_numpy(state).unsqueeze(0)
        action_probs = self.actor_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()

        return action.item(), log_prob, entropy

        
        
    def learn(self, next_state):
        """
        Aprendizaje del agente basado en las experiencias almacenadas.

        Parámetros:
            next_state: El siguiente estado del entorno para calcular el retorno.
        """
        next_state = torch.from_numpy(next_state).unsqueeze(0)
        next_value = self.critic_net(next_state)

        returns = self.compute_returns(next_value, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        """
        Limpia la memoria del agente.
        """
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_value, gamma=0.99):
        """
        Calcula los retornos ajustados para cada paso.

        Parámetros:
            next_value: Valor estimado del siguiente estado.
            gamma (float): Factor de descuento.

        Retorna:
            Lista de retornos ajustados.
        """
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns