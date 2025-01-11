#####################
### Bibliothèques ###
#####################
import json
import pandas as pd
import numpy as np
import chess
import random

from Fct.fct_preprocess import *
from Fct.fct_eval import *

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam


##############################
### Chargement des données ###
##############################
# Chargement du modèle
model = load_model("Models\Modele_1_TF_25EPOCHS.keras")

# Chargement du dictionnaire de traduction coup / integer pour faire la traduction entre le model et le chess.board
with open('Models/move_int_dico.json', 'r') as file:
    move_int = json.load(file)
move_int_dico = {v: k for k, v in move_int.items()}

########################
### Faire jouer l'IA ###
########################
# Faire jouer une partie de l'IA contre elle-même ou contre un joueur "random"
play_game(model,"random",move_int_dico,print_game = True)

# Faire jouer plusieurs partie de l'IA contre elle-même ou contre un joueur "random"
play_multiple_games(model, "random", move_int_dico, 50)

# Faire contre l'IA contre un humain
play_human_vs_ia(model, move_int_dico, human_color='white')