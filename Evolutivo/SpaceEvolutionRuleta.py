import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import cv2
import os



env = gym.make('ALE/SpaceInvaders-v5', obs_type='grayscale')

# estructura de la red neuronal con PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        logits = self.linear_layers(x)
        return logits
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename))
        else:
            print(f"No se encontró el archivo: {filename}")

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

#Crear la población inicial de agentes Cambiar el valor de la poblacion inicial para entrenar mas diverso, pero aumenta considerablemente el tiempo de entrenamiento y de memoria.
population_size = 50  
population = [NeuralNetwork().apply(init_weights) for _ in range(population_size)]

#Cargar agente entrenado
for i in range(population_size):
    filename = f"agent_{i}.pth"
    if os.path.exists(filename):
        population[i].load_state_dict(torch.load(filename))


def preprocess_state(state):
    # Redimensionar la imagen y Normalizar la imagen
    resized = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    # Añadir una dimensión para los canales y otra para el batch
    processed_state = np.expand_dims(normalized, axis=0)
    processed_state = np.expand_dims(processed_state, axis=0)  # Añadir dimensión de canales
    return processed_state


#función de evaluación para los agentes - Funcion fitness
def evaluate_agent(agent, env, render=False):
    total_reward = 0
    done = False
    obs = env.reset()

    while not done:
        if render:
            env.render() 
        # Extraer solo la imagen de la tupla
        image = obs[0] 
        # Preprocesar la imagen
        processed_image = preprocess_state(image)

        # Convertir a tensor de PyTorch
        obs_tensor = torch.from_numpy(processed_image).float()

        # Usar la red neuronal para decidir la acción
        logits = agent(obs_tensor)
        action = torch.argmax(logits, dim=1).item()

        # Ejecutar la acción en el entorno
        ####obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)[:4]
        total_reward += reward

    return total_reward

# Elige los siguientes padres con el peso de su puntaje como probabilidad
def select_parents_roulette(population, scores, num_parents):
    # Arreglo para simular probabilidades al escoger
    scores_sum = sum(scores)
    probs = [score / scores_sum for score in scores]
    # Elegir indices basados en esta probabilidad
    parent_indices = np.random.choice(len(population), size=num_parents, replace=False, p=probs)
    # Guarda padre basado en los indices anteriores
    parents = [population[idx] for idx in parent_indices]
    
    return parents


def crossover(parent1, parent2):
    child = NeuralNetwork().apply(init_weights)

    # Obtener solo las capas lineales de los padres
    parent1_linear_layers = [module for module in parent1.modules() if isinstance(module, nn.Linear)]
    parent2_linear_layers = [module for module in parent2.modules() if isinstance(module, nn.Linear)]

    # Elegir un punto de cruce al azar
    crossover_idx = random.randint(0, len(parent1_linear_layers)-1)

    # Combinar los pesos de las capas lineales de los padres en el hijo
    child_linear_layers = [module for module in child.modules() if isinstance(module, nn.Linear)]
    for i in range(len(child_linear_layers)):
        if i <= crossover_idx:
            child_linear_layers[i].weight.data = parent1_linear_layers[i].weight.data.clone()
            child_linear_layers[i].bias.data = parent1_linear_layers[i].bias.data.clone()
        else:
            child_linear_layers[i].weight.data = parent2_linear_layers[i].weight.data.clone()
            child_linear_layers[i].bias.data = parent2_linear_layers[i].bias.data.clone()

    return child



initial_mutation_power = 1  # Tasa de mutación más alta al principio
final_mutation_power = 0.1  # Tasa de mutación más baja hacia el final

def mutate(agent):

    mutation_power =    initial_mutation_power - ((initial_mutation_power - final_mutation_power) * generation / num_generations) #Con esta linea la mutacion se va reduciendo con el pasar de las generaciones.
    for param in agent.linear_layers.parameters():
        if len(param.shape) > 1:  # Solo alterar los pesos, no los sesgos
            param.data += mutation_power * torch.randn_like(param)




#ciclo principal del algoritmo evolutivo
num_generations = 50 #Cantidad de veces que se entrenará la población. Cambiar este valor para entrenar mas generaciones
num_parents = int(population_size / 2)
num_children = population_size - num_parents

for generation in range(num_generations):
    # Evaluar todos los agentes
    scores = [evaluate_agent(agent, env, render = False) for agent in population]
    parents = select_parents_roulette(population, scores, num_parents)

    # Generar descendencia
    children = []
    for _ in range(num_children // 2):  # Dividir por 2 para obtener pares de padres
        parent1, parent2 = random.sample(parents, 2)
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        children.extend([child1, child2])

    # Aplicar mutación
    for child in children:
        mutate(child)

    # Crear la nueva generación
    population = parents + children
    print(f"Generación {generation}, Mejor puntuación: {max(scores)}")

# Después del ciclo de entrenamiento se guardan los agentes
for i, agent in enumerate(population):
    torch.save(agent.state_dict(), f"agent_{i}.pth")

#for i, agent in enumerate(population):
#    filename = f"agent_{i}.pth"
#    agent.save(filename)

env_render = gym.make('ALE/SpaceInvaders-v5', obs_type='grayscale', render_mode='human')

best_agent = max(zip(population, scores), key=lambda x: x[1])[0]
evaluate_agent(best_agent, env_render, render=True)

# Cerrar el entorno
env.close()
env_render.close()