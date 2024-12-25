# С использованием билиотеки OpenCv
import matplotlib.pyplot as plt
import numpy as np
import cv2

original_image = cv2.imread('cars.jpg')
img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB) # Преобразует из BGR в RGB


twoDimageimg = img.reshape((-1,3))
twoDimageimg = np.float32(twoDimageimg)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # Остановка алгоритма при выполнении одного из условий 
K = 4
attempts=10

ret,label,center=cv2.kmeans(twoDimageimg,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center) 
res = center[label.flatten()] #  Заменяет каждую метку пикселя в массиве label на соответствующий цвет центра кластера 
result_image = res.reshape((img.shape))


# Вывод изображения оригинального
plt.figure() 
plt.imshow(img)
plt.title('Оригинальное изображение')
plt.axis('off')
plt.show()

# Отображение сегментированного изображения
plt.figure() 
plt.imshow(result_image)
plt.title(f'Сегментированное изображение (k={K})')
plt.axis('off')
plt.show()