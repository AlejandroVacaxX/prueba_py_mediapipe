import cv2
import numpy as np
# Leer la imagen
#img = cv2.imread('tiburon.jpg')
cap = cv2.VideoCapture(1)
ret, img = cap.read()

# 1. Crear una imagen de fondo (negra)
#img = np.zeros((500, 500, 3), dtype="uint8")
h, w, _ = img.shape

# 2. Configurar el texto
texto = "Te estas quedando Dormido"
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 10
thickness = 5
color = (0, 0, 255) # Blanco

# 3. Calcular el tamaño del texto
text_size = cv2.getTextSize(texto, font, fontScale, thickness)[0]
text_width, text_height = text_size

# 4. Calcular coordenadas de inicio (x, y) para centrar
x = (w - text_width) // 2
y = (h + text_height) // 2

# 5. Poner el texto en la imagen
cv2.putText(img, texto, (x, y), font, fontScale, color, thickness, cv2.LINE_AA)

# Mostrar resultado
cv2.imshow("Texto Centrado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

