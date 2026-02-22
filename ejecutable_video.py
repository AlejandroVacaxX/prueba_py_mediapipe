# BORRA LO QUE TENGAS Y PON ESTO:
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2

# Acceder a solutions via mp.solutions (funciona en 0.10.x)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# --- 1. CONFIGURACIÓN DEL MODELO ---
# Asegúrate de que 'face_landmarker.task' esté en la carpeta
model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configurar opciones: MODO VIDEO es vital para webcam
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # le decimos que no seran fotos, si no video en vivo
    output_face_blendshapes=True,  # ¡Importante para detectar sueño!
    output_facial_transformation_matrixes=True,
    num_faces=1,
)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        # --- ARREGLO 1: Conversión de Formato (Evita que se cierre al detectar cara) ---
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        # Dibujar malla facial
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        # Dibujar contornos
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    return annotated_image


# --- 3. BUCLE PRINCIPAL (MAIN LOOP) ---
def main():
    # Inicializar la cámara (0 suele ser la webcam por defecto)
    cap = cv2.VideoCapture(1)

    # Crear el detector dentro de un bloque 'with' para asegurar que se cierre bien
    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Iniciando cámara... presiona 'q' para salir.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignorando frame vacío.")
                continue
            """
            Dimenciones del frame
            Definire las dimensiones para centrar texto
            """

            height, width, _ = frame.shape

            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 5.5
            ancho_font = 3
            color_text = (0, 255, 0)  # verde al inicio
            texto_en_pantalla = "Conduccion Segura"
            text_size = cv2.getTextSize(
                texto_en_pantalla, font, font_scale, ancho_font
            )[0]
            text_width, text_height = text_size

            # Calculamos x,y
            x = (width - text_width) // 2
            y = (
                height - text_height
            ) // 4  # cambia el numero entre que se divide para subir o bajar el texto

            # A. PREPARAR LA IMAGEN
            # OpenCV usa BGR, MediaPipe necesita RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # B. CALCULAR TIMESTAMP (Requerido para modo VIDEO)
            # Usamos el tiempo actual en milisegundos
            frame_timestamp_ms = int(time.time() * 1000)

            # C. DETECTAR
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # D. VISUALIZAR RESULTADOS
            # Convertimos de vuelta a BGR para mostrarlo en OpenCV
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Logica de mimir
            if detection_result.face_blendshapes:
                # Accedemos a la primera cara detectada ([0])
                face_blendshapes = detection_result.face_blendshapes[0]  # izquierdo

                """ Imprimimos todos los blendshapes para ver sus nombres y valores 
                for blendshape in face_blendshapes:
                    print(f"{blendshape.category_name}: {blendshape.score:.3f}")
                """

                # Definimos las variables para ambos ojos
                ojo_cerrado_Izquierdo = 0.0
                ojo_cerrado_Derecho = 0.0
                boca_abierta = 0.0
                # ambos_ojos_cerrados = False

                for blendshape in face_blendshapes:
                    if blendshape.category_name == "eyeBlinkLeft":
                        ojo_cerrado_Izquierdo = blendshape.score
                    elif blendshape.category_name == "eyeBlinkRight":
                        ojo_cerrado_Derecho = blendshape.score
                    if blendshape.category_name == "jawOpen":
                        boca_abierta = blendshape.score

                if ojo_cerrado_Izquierdo > 0.500 and ojo_cerrado_Derecho > 0.500:

                    texto_en_pantalla = "       Zzzzz.."
                    color_text = (0, 0, 255)  # color rojo
                    print(
                        f"Alerta de ojos Cerrados {ojo_cerrado_Derecho:.3f} {ojo_cerrado_Izquierdo:.3f}"
                    )

                if boca_abierta > 0.500:
                    texto_en_pantalla = "Bostezando"
                    color_text = (255, 0, 0)  # color azul?
                    print(f"Conductor con Boca abierta {boca_abierta:3f}")

                # esto revisaba ojo por ojo pero ya no lo necesito
                """
                if ojo_cerrado_Izquierdo > 0.500:
                    texto_en_pantalla = 'Ojo Izquierdo'
                    color_text = (0, 0, 255)
                    print(f"Alerta de Ojo Izquierdo Cerrado {ojo_cerrado_Izquierdo:.3f} ")
                elif ojo_cerrado_Derecho > 0.500:
                    texto_en_pantalla = 'Ojo Derecho'
                    color_text = (0, 0, 255)
                    print(f"Alerta de Ojo Derecho Cerrado {ojo_cerrado_Derecho:.3f} ")
                """

                cv2.putText(
                    frame,
                    texto_en_pantalla,
                    (x, y),
                    font,
                    font_scale,
                    color_text,
                    ancho_font,
                    cv2.LINE_AA,
                )

            # Mostrar en ventana
            # cambiar bgr por 'frame' cuando quiera dejar de mostrar la malla y solo el texto
            cv2.imshow("Detector de Sueño UAM", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
