#Без использования библиотеки 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def kmeans_segmentation(image_path, k, max_iterations):
    image = np.array(Image.open(image_path))
    plt.figure()
    plt.imshow(image)
    plt.title('Оригинальное изображение')
    plt.axis('off')
    plt.show()
    
    pixels = image.reshape((-1, 3)) # Преобразование изображения в двумерный массив
    pixels = np.float32(pixels)
    
    centers = pixels[np.random.choice(pixels.shape[0], k, replace=False)] # Инициализация случайных центров кластеров

    for _ in range(max_iterations):
        
        distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2) # Рассчитываем расстояние между каждым пикселем и центрами кластеров
        labels = np.argmin(distances, axis=1) # Определяем ближайший центр для каждого пикселя
        new_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)]) # Обновляем центры кластеров как среднее значение пикселей в каждом кластере

        if np.all(centers == new_centers): # Если центры кластеров перестали изменяться, выходим из цикла
            break
        centers = new_centers

    res = center[label.flatten()] #  Заменяет каждую метку пикселя в массиве label на соответствующий цвет центра кластера 
    segmented_image = res.reshape((img.shape))
    return segmented_image

image_path = "cars.jpg"
k = 4

segmented_image = kmeans_segmentation(image_path, k=k, max_iterations=100)

plt.figure()
plt.imshow(segmented_image.astype(np.uint8))
plt.title(f'Сегментированное изображение (k={k})')
plt.axis('off')
plt.show()