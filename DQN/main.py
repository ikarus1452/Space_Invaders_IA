import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
from random import randint
import pandas as pd
import sys
sys.path.append('../')
from dqn_cnn import DQNCnn
from dqn_agent import DQNAgent
from stack_frame import preprocess_frame, stack_frame

env = gym.make('ALE/SpaceInvaders-v5')
env.seed(0)

# si se va a usar la gpu o cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames
    


INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99  # O cualquier otro valor
BUFFER_SIZE = 100000  # O cualquier otro valor mayor
BATCH_SIZE = 64        # tamaño del batch
LR = 0.00001  # O cualquier otro valor menor
TAU = 0.05             # actualizacion de parametros
UPDATE_EVERY = 100      # frecuencia de actualizacion 
UPDATE_TARGET = 2000  # O cualquier otro valor
EPS_START = 1.0        # valor inicial de epsilon
EPS_END = 0.01       # valor final de epsilon
EPS_DECAY = 1000  # O cualquier otro valor
start_epoch = 0
arq = 2  
agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn,arq)
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)


'''ver jugar al agente de forma aleatoria '''
def random_play():
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='human') 
    score = 0
    state, _ = env.reset()
    while True:
        env.render()
        action = randint(0,3)
        state, reward, done, truncated, _ = env.step(action)
        score += reward
        if done or truncated:
            state, _ = env.reset()
            print("Your Score at end of game is: ", score)
            break
    env.close()

'''graficar los resultados'''
def test_graph(df): 
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
    plt.title('Scores obtenidos por episodio DQN')
    plt.legend()  # Mostrar leyenda
    plt.show()

'''entrenar al agente'''
df = pd.DataFrame(columns=['n_episodio', 'score'])
def train(n_episodes): 
    agent.load()
    for i_episode in range(start_epoch + 1, n_episodes+1):
        reset, _ = env.reset()
        state = stack_frames(None, reset, True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        nuevo_dato = [i_episode,score]    
        df.loc[len(df)] = nuevo_dato
    test_graph(df)
    agent.save()
    env.close()     

''' cargar el modelo y ver jugar al agente'''
def test(n_episodes): 
    #env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    score = 0
    agent.load()
    for i_episode in range(start_epoch + 1, n_episodes+1):
        reset, _ = env.reset()
        state = stack_frames(None, reset, True)
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, truncated, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            state = next_state
            if done or truncated:
                break
        env.close()
    return score/n_episodes


train(300)
score_prom = test(10)
print (score_prom)
print(df)

