import cv2
import numpy as np
import matplotlib.pyplot as plt

def aplicar_grabcut(ruta_imagen, rect, iteraciones=5):
    img = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iteraciones, cv2.GC_INIT_WITH_RECT)

    mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    resultado = img_rgb * mask_final[:, :, np.newaxis]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagen original")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_final, cmap='gray')
    plt.title("Máscara final")

    plt.subplot(1, 3, 3)
    plt.imshow(resultado)
    plt.title("Segmentación GrabCut")

    plt.show()
