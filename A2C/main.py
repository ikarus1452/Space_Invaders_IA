import time
import gymnasium as gym
import random
import torch
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import os

import sys

from agente import A2CAgent
from actor_critico import ActorCnn, CriticCnn
from stack_frame import preprocess_frame, stack_frame


env = gym.make('ALE/SpaceInvaders-v5')
env.seed(0)




# Función para apilar cuadros del juego y crear un estado compuesto
def stack_frames(frames, state, is_new=False):
    """
    Procesa y apila cuadros para crear un estado compuesto.

    Parámetros:
        frames: Cuadros previamente apilados.
        state: Estado actual a ser procesado y apilado.
        is_new (bool): Indica si se trata de un nuevo episodio.

    Retorna:
        Cuadros apilados actualizados.
    """
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames



#crear el agente

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.96           # discount factor 0.96 para fomentar una mayor consideración de las recompensas inmediatas.
ALPHA= 0.0002          # Actor learning rate
BETA = 0.001          # Critic learning rate
UPDATE_EVERY = 50    # Cada cuantos episodios se actualiza la red
start_epoch = 0
agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)


#Entrenar el agente



def export_data(df):
    """
    Exporta datos de entrenamiento a un archivo CSV.

    Parámetros:
        df (DataFrame): Datos de entrenamiento a exportar.
    """
    # Crear la carpeta si no existe
    if not os.path.exists('Data'):
        os.makedirs('Data')
    # Obtener la ruta completa del archivo CSV
    file_path = os.path.join('Data', 'Data_A2C')
    # Exportar los datos del DataFrame a un archivo CSV (modo de escritura "w")
    df.to_csv(file_path, index=False, mode='w')


'''graficar los resultados'''
def test_graph(df): 
    """
    Grafica los resultados del entrenamiento.

    Parámetros:
        df (DataFrame): Datos de entrenamiento.
    """
    clear_output(True)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df.plot(x='n_episodio', y='score', kind='line', ax=ax, label='Score')  # Trazar puntos de datos como dispersión

    # Calcular la línea de regresión utilizando numpy
    x = df['n_episodio']
    y = df['score']
    coeffs = np.polyfit(x, y, 4)  # Ajustar una línea de regresión de grado 4 (una recta)
    poly_eq = np.poly1d(coeffs)  # Ecuación de la línea de regresión
    x_fit = np.linspace(x.min(), x.max(), 100)  # Valores x para la línea de regresión
    y_fit = poly_eq(x_fit)  # Valores y para la línea de regresión

    plt.plot(x_fit, y_fit, 'r-', label='Curva de regresión ')  # Trazar la línea de regresión en rojo ('r-')
    plt.ylabel('Score')
    plt.xlabel('Episodio')
    plt.title('Scores obtenidos por episodio A2C')
    plt.legend()  # Mostrar leyenda
    plt.show()



df = pd.DataFrame(columns=['n_episodio', 'score'])
def train(n_episodes): 
    """
    Entrena el agente A2C.

    Parámetros:
        n_episodes (int): Número de episodios para el entrenamiento.
    """
    
    agent.actor_net.load()  #Descomentar estas dos lineas de codigo para cargar los pesos de la red
    agent.critic_net.load()

    for i_episode in range(start_epoch + 1, n_episodes+1):
        reset, _ = env.reset()
        state = stack_frames(None, reset, True)
        score = 0
        
        while True:
            
            action, log_prob, entropy = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, log_prob, entropy, reward, done, next_state)
            state = next_state
            if done:
                break
        
        nuevo_dato = [i_episode,score]    
        df.loc[len(df)] = nuevo_dato
        print(f"Episodio {i_episode}/{n_episodes} - Score: {score}")
        
    
    agent.actor_net.save()
    agent.critic_net.save()

    env.close()

def render_game(n_episodes):
    """
    Renderiza y visualiza el juego.

    Parámetros:
        n_episodes (int): Número de episodios para visualizar.
    """
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    for i_episode in range(1, n_episodes + 1):
        initial_state = env.reset()
        # Asumiendo que el estado inicial es el primer elemento de la tupla
        state_image = initial_state[0] if isinstance(initial_state, tuple) else initial_state
        state = stack_frames(None, state_image, True)
        total_reward = 0
        done = False

        while not done:
            env.render()  # Renderiza el entorno para visualizar el juego
            action, _, _ = agent.act(state)  # Elige una acción basada en la política aprendida
            # Asegúrate de obtener correctamente el próximo estado del juego de env.step()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward  # Acumula la recompensa
            next_state = stack_frames(state, next_state, False)  # Apila el siguiente estado
            state = next_state  # Actualiza el estado

        print(f"Episodio: {i_episode}, Total Reward: {total_reward}")

    env.close()  # Cierra el entorno una vez finalizados los episodios



train(1)
test_graph(df)
export_data(df)
render_game(10)