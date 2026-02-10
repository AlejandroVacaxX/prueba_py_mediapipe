import sys
print(f"Python version: {sys.version}")

try:
    print("1. Intentando importar mediapipe...")
    import mediapipe
    print(f"   -> Éxito. Ruta: {mediapipe.__file__}")
    
    print("2. Intentando importar opencv...")
    import cv2
    print(f"   -> Éxito. Versión: {cv2.__version__}")

    print("3. Intentando importar solutions MANUALMENTE...")
    # Esto forzará a que salga el error real oculto
    from mediapipe.python import solutions
    print("   -> ¡Éxito! solutions cargó correctamente.")

except ImportError as e:
    print(f"\n❌ ERROR CRÍTICO DE IMPORTACIÓN:\n{e}")
except AttributeError as e:
    print(f"\n❌ ERROR DE ATRIBUTO (Probable instalación corrupta):\n{e}")
except Exception as e:
    print(f"\n❌ OTRO ERROR:\n{e}")