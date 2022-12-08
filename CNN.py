import keras as Kconda
import tensorflow as tf
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando a Segunda Camada de Convolução e Pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('datasets/cachorro_e_gato/dataset_treino',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_set = validation_datagen.flow_from_directory('datasets/cachorro_e_gato/dataset_validation',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

# Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=5,
                         validation_data=validation_set,
                         validation_steps=2000)



cachorroegato = 100
covid = 4
bartehomer = 20

for i in range(cachorroegato): # adaptar para cada base

    # gato e cachorro
    image_number = 2150
    image_path = 'datasets/cachorro_e_gato/dataset_teste/' + str(image_number) + '.jpg'
    image_number += 10

    # bart e homer
    # image_number = 1
    # image_path = 'datasets/homer_e_bart/dataset_teste/bart/bart' + str(image_number) + '.bmp'
    # image_path2 = 'datasets/homer_e_bart/dataset_teste/homer/homer' + str(image_number) + '.bmp'

    # covid
    # image_number = 0
    # image_number2 = 35
    # image_path2 = 'datasets/xray_covid/dataset_teste/PNEUMONIA/streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day'  + str(image_number) + '.jpg'
    # image_path = 'datasets/cachorro_e_gato/dataset_teste/NORMAL/NORMAL2-IM-00' + str(image_number2) + '-0001.jpeg' 
    # image_number += 1
    # image_number2 += 1

    test_image = image.load_img(image_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices

    if result[0][0] == 1:
        prediction = 'Cachorro'
    else:
        prediction = 'Gato'

    Image(filename=image_path)

    #     test_image = image.load_img(image_path2, target_size = (64, 64))
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis = 0)
    # result = classifier.predict(test_image)
    # training_set.class_indices

    # if result[0][0] == 1:
    #     prediction = 'Cachorro'
    # else:
    #     prediction = 'Gato'

    # Image(filename=image_path2)