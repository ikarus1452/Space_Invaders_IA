import numpy as np
import cv2

def preprocess_frame(screen, exclude, output):
    """
    Preprocesa un cuadro (frame) del juego convirtiéndolo a escala de grises, recortándolo y redimensionándolo.

    Parámetros:
        screen (array): Imagen RGB del cuadro del juego.
        exclude (tuple): Coordenadas para recortar la imagen (ARRIBA, DERECHA, ABAJO, IZQUIERDA).
        output (int): Tamaño de la imagen de salida (tanto para altura como para anchura).

    Retorna:
        array: Cuadro preprocesado.
    """
    
    # convertir la imagen a una escala de grises

    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
    #partir la pantalla
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    
    # convertir a float y normalizar
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    # ajustar la imagen a 84 x 84
    screen = cv2.resize(screen, (output, output), interpolation = cv2.INTER_AREA)
    return screen

def stack_frame(stacked_frames, frame, is_new):
    """
    Apila cuadros para crear un estado compuesto por varios cuadros consecutivos.

    Parámetros:
        stacked_frames (array): Arreglo de cuadros previamente apilados.
        frame: Cuadro actual preprocesado que se añadirá al conjunto apilado.
        is_new (bool): Indica si se trata de un nuevo episodio.

    Retorna:
        array: Conjunto actualizado de cuadros apilados.
    """
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames