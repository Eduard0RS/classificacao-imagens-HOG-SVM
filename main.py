import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog

dataset_dir = "Car-Bike-Dataset"

# Lista para armazenar as imagens e rótulos
images = []
labels = []

# Definir o tamanho desejado para redimensionar as imagens
target_size = (64,64)

# Percorrer as subpastas "Bike" e "Car"
for subdir in os.listdir(dataset_dir):
    subpath = os.path.join(dataset_dir, subdir)
    if os.path.isdir(subpath):
        label = subdir.lower()  # Rótulo será "bike" ou "car"
        for filename in os.listdir(subpath):
            image_path = os.path.join(subpath, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Converter a imagem para escala de cinza
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Redimensionar a imagem para o tamanho desejado
                resized_image = cv2.resize(gray_image, target_size)
                images.append(resized_image)
                labels.append(label)

# Dividir os dados em conjuntos de treinamento e teste com 70-30
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,stratify=labels)

# Função para calcular o HOG de uma imagem
def calculate_hog(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return hog_features

# Calcular o HOG para todas as imagens de treinamento e teste
train_hog_features = [calculate_hog(image) for image in train_images]
test_hog_features = [calculate_hog(image) for image in test_images]

# Criar um objeto de classificador SVM
svm_classifier = SVC()

# Treinar o classificador SVM com as imagens de treinamento e seus rótulos correspondentes
svm_classifier.fit(train_hog_features, train_labels)

# Fazer a classificação das imagens de teste usando o SVM treinado
predicted_labels = svm_classifier.predict(test_hog_features)

# Avaliar a precisão do modelo
accuracy = np.mean(predicted_labels == test_labels)
print("Acurácia do modelo: {:.2f}%".format(accuracy * 100))

