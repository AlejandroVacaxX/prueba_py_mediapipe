# BORRA LO QUE TENGAS Y PON ESTO:
import cv2
import mediapipe as mp
import numpy as np
import time


# Importamos explícitamente las soluciones para forzar a Python a cargarlas
from mediapipe.solutions import drawing_utils
from mediapipe.solutions import drawing_styles
from mediapipe.solutions import face_mesh

# Configuración de variables (opcional, si las usabas con 'mp.')
mp_drawing = drawing_utils
mp_face_mesh = face_mesh
mp.framework.formats.landmark_pb2 # <--- Agrega este import arriba

# --- 1. CONFIGURACIÓN DEL MODELO ---
# Asegúrate de que 'face_landmarker.task' esté en la carpeta
model_path = 'face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode 

# Configurar opciones: MODO VIDEO es vital para webcam
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # le decimos que no seran fotos, si no video en vivo
    output_face_blendshapes=True, # ¡Importante para detectar sueño!
    output_facial_transformation_matrixes=True,
    num_faces=1
)



def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks 
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        # --- ARREGLO 1: Conversión de Formato (Evita que se cierre al detectar cara) ---
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])

        # --- ARREGLO 2: Ruta Correcta (mp.solutions en vez de mp.python.solutions) ---
        
        # Dibujar malla facial
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, # <--- AQUÍ ESTABA EL ERROR
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        
        # Dibujar contornos
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS, # <--- AQUÍ TAMBIÉN
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

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
            
    
            # E. LÓGICA DE DORMIR
            if detection_result.face_blendshapes:
                # Accedemos a la primera cara detectada ([0])
                face_blendshapes = detection_result.face_blendshapes[0] #izquierdo
                
                # Imprimimos todos los blendshapes para ver sus nombres y valores /*
               #for blendshape in face_blendshapes:
            #      print(f"{blendshape.category_name}: {blendshape.score:.3f}")
                #
                #Definimos las variables para ambos ojos

                ojo_cerrado_Izquierdo = 0.0
                ojo_cerrado_Derecho = 0.0
                for blendshape in face_blendshapes:
                    if blendshape.category_name == 'eyeBlinkLeft':
                        ojo_cerrado_Izquierdo = blendshape.score
                    elif blendshape.category_name == 'eyeBlinkRight':
                        ojo_cerrado_Derecho = blendshape.score
                    
                texto_en_pantalla = 'Correcto'
                color_text = (0,255,0) #verde
                if ojo_cerrado_Izquierdo > 0.500 and ojo_cerrado_Derecho > 0.500:
                    texto_en_pantalla = 'Ojos Cerrados'
                    color_text = (0, 0, 255)
                    print(f"Alerta de ojos Cerrados {ojo_cerrado_Derecho:.3f} {ojo_cerrado_Izquierdo:.3f}")
                if ojo_cerrado_Izquierdo > 0.500:
                    texto_en_pantalla = 'Ojo Izquierdo'
                    color_text = (0, 0, 255)
                    print(f"Alerta de Ojo Izquierdo Cerrado {ojo_cerrado_Izquierdo:.3f} ")
                elif ojo_cerrado_Derecho > 0.500:
                    texto_en_pantalla = 'Ojo Derecho'
                    color_text = (0, 0, 255)
                    print(f"Alerta de Ojo Derecho Cerrado {ojo_cerrado_Derecho:.3f} ")

                    
                cv2.putText(bgr_annotated_image, texto_en_pantalla, (50,50),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color_text, 2)    
                    
            
                



            # Mostrar en ventana
            cv2.imshow('Detector de Sueño UAM', bgr_annotated_image)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()