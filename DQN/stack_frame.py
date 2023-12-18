import numpy as np
import cv2

def preprocess_frame(screen, exclude, output):
        # Parametros
          #  screen (array): imagen RGB
          #  exclude (tuple): Sección a recortar (ARRIBA, DERECHA, ABAJO, IZQUIERDA)
          #  output (int): tamaño de la imagen de salida
    # Convertir la imagen a una escala de grises
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
    # Partir la pantalla
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    
    # Convertir a float y normalizar
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    # Ajustar la imagen a 84 x 84
    screen = cv2.resize(screen, (output, output), interpolation = cv2.INTER_AREA)
    return screen

def stack_frame(stacked_frames, frame, is_new):        
        # Parametros
           # stacked_frames (array): Cuatro frames apilados
           # frame: Cuadro preprocesado 
           # is_new: si es el primer frame
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames