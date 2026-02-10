import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# 1. Importar la lógica de Tareas (Nueva API)
from mediapipe.tasks.python import vision

# 2. Importar las herramientas de dibujo (API Clásica)
# CAMBIO CLAVE: Importamos directamente desde el submódulo para evitar el error de atributo
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import drawing_styles

# Definimos los estilos de conexión manualmente si vision no los trae
# (Esto asegura compatibilidad si mezclas versiones)
ConnectionDrawingSpec = drawing_utils.DrawingSpec


def draw_landmarks_on_image(rgb_image, detection_result):
    """ 
    Dibuja la malla facial sobre la imagen original.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Iterar sobre las caras detectadas
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Dibuja la malla (Tesselation)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        # Dibuja los contornos (Cara, ojos, labios)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )

        # Dibuja los iris (Opcional, pero se ve genial)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """
    Grafica los coeficientes de expresión facial.
    Útil para depurar, pero para tu app usarás los valores crudos.
    """
    # Extraer nombres y puntuaciones
    face_blendshapes_names = [category.category_name for category in face_blendshapes]
    face_blendshapes_scores = [category.score for category in face_blendshapes]

    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
        face_blendshapes_ranks,
        face_blendshapes_scores,
        label=[str(x) for x in face_blendshapes_ranks],
    )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(
            patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top"
        )

    ax.set_xlabel("Puntuación")
    ax.set_title("Blendshapes (Expresiones Faciales)")
    plt.tight_layout()
    plt.show()
