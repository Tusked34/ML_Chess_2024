#####################
### Bibliothèques ###
#####################
import pandas as pd
import json
import chess
import numpy as np
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from Fct.fct_preprocess import *

###############################
### Chargement des Données ####
###############################
data = pd.read_csv('data\data_cleaned.csv' )

print("Chargement des données OK \n")

X = data['fen']
y = data['best_move_m1']

print("Split des variables en X et y OK \n")

######################################
### Traitement finale des données ####
######################################

X = X.apply(fen_to_matrix)
X = np.array(X.tolist())

print("Transformation du FEN en matrice numpy OK \n")

# Encodage les mouvements et conversion en catégories
y, move_to_int = encode_moves(y)
y = to_categorical(y, num_classes=len(move_to_int))

print("Encodage des coups OK \n")

# Sauvegarde de move_to_int dans un fichier JSON
with open('Models\move_int_dico.json', 'w') as file:
    json.dump(move_to_int, file)

print("Sauvegarde du dico Coups/Integer OK \n")

############################
### Definition du modele ###
############################

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(13, 8, 8)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(move_to_int), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("Definition du modele OK \n")

####################
### Entrainement ###
####################
model.save("Models\Modele_1_TF_50EPOCHS_1Mdata.keras")
print("Sauvegarde du modele OK \n")