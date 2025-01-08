
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

import warnings
warnings.filterwarnings("ignore")

def extract_data(file_path, max_rows=50):
    """
    Extrait les données d'un fichier JSONL contenant des positions d'échecs et les évalue, 
    en créant un DataFrame à partir des informations extraites.

    Args:
        file_path (str): Chemin du fichier contenant les données au format JSONL (une position par ligne).
        max_rows (int, optional): Nombre maximal de lignes à lire dans le fichier. Par défaut, 50.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les colonnes suivantes :
            - "fen" : Représentation FEN de l'échiquier.
            - "knodes" : Nombre de nœuds évalués.
            - "depth" : Profondeur de recherche de l'évaluation.
            - "cp" : Centipawn score de l'évaluation (avantage positionnel en valeurs de pions).
            - "mate" : Nombre de coups avant échec et mat (si applicable).
            - "line" : Ligne de coups prévus.
    """
    data = []  # Liste pour stocker les données extraites
    
    # Ouvrir le fichier en lecture
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            # Arrêter la lecture si le nombre maximal de lignes est atteint
            if idx >= max_rows:
                break

            # Charger la ligne JSON en dictionnaire Python
            position = json.loads(line)
            fen = position.get("fen")  # Extraire le code FEN
            
            # Parcourir les évaluations disponibles pour cette position
            for eval_data in position.get("evals", []):
                knodes = eval_data.get("knodes")  # Nombre de nœuds évalués
                depth = eval_data.get("depth")   # Profondeur de recherche
                
                # Parcourir les variantes principales (principal variations)
                for pv in eval_data.get("pvs", []):
                    cp = pv.get("cp")       # Score centipawn
                    mate = pv.get("mate")   # Nombre de coups avant mat (si applicable)
                    line = pv.get("line")   # Ligne de coups prévus
                    
                    # Ajouter les données extraites à la liste
                    data.append({
                        "fen": fen,
                        "knodes": knodes,
                        "depth": depth,
                        "cp": cp,
                        "mate": mate,
                        "line": line
                    })

    # Convertir la liste de dictionnaires en DataFrame pandas
    return pd.DataFrame(data)

def extract_next_move(df):
    """
    Extrait le premier coup de la colonne 'line' et le stocke dans 'best_move_m1'.
    """
    df["best_move_m1"] = df["line"].str.extract(r'(\S+)', expand=False)
    return df

def most_popular_predict_V1(df):
    """
    Ajoute la colonne 'best_move_m2' avec le coup le plus fréquent pour chaque position FEN.
    """
    most_popular_moves = (
        df.groupby('fen')['best_move_m1']
        .agg(lambda x: x.value_counts().idxmax())  # Coup le plus fréquent par groupe FEN
        .rename("best_move_m2"))
    
    df = df.merge(most_popular_moves, on='fen', how='left')
    return df

def filter_best_predict(df):
    """
    Filtre les meilleures évaluations par position FEN, en priorisant la profondeur maximale et les évaluations de mat.
    """
    filtered_rows = []

    # Grouper par position FEN
    for fen, group in df.groupby('fen'):
        # Étape 1 : Sélectionner les évaluations avec la profondeur maximale
        max_depth = group['depth'].max()
        best_evals = group[group['depth'] == max_depth]
        
        # Étape 2 : Priorité à une évaluation de mat si elle existe
        if 'mate' in best_evals and best_evals['mate'].notna().any():
            best_row = best_evals.loc[best_evals['mate'].idxmin()]
        else:
            # Sinon, maximiser la valeur `cp`
            best_row = best_evals.loc[best_evals['cp'].idxmax()]
        
        # Ajouter la meilleure évaluation à la liste
        filtered_rows.append(best_row)
    
    # Créer un nouveau DataFrame avec les meilleures évaluations
    return pd.DataFrame(filtered_rows).reset_index(drop=True)

def drop_usuless_columns(df) :
    """
    Supprime les colonnes inutiles du DataFrame.
    """
    df = df.drop(columns=['line','mate','cp','depth','knodes'])
    return df

def fen_to_matrix(fen: str):
    """
    Convertit une position d'échecs en FEN en une matrice 8x8x13.
    8*8 pour les cases de l'échequier et *12 pour chaque pieces uniques (6 Blanches et 6 Noires)
    
    Parameters: 
        La représentation FEN de la position.
    
    Returns:
        Une matrice numpy 8x8x12 représentant la position.
    """
    board = chess.Board(fen) #  convertit une représentation FEN en un objet manipulable qui contient toutes les informations sur la position d'échecs. Cet objet permet ensuite de travailler directement avec l'échiquier dans le code.
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def encode_moves(moves):
    """
        Cette fonction prend une liste de coups d'échecs (sous forme de chaînes) et encode chaque coup unique
        sous forme d'un entier unique. Elle retourne également un dictionnaire qui associe chaque coup à son
        entier correspondant.
    """
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int

def predict_next_move(board, model, move_int_dico):
    """
    Prédit les prochains coups dans une partie d'échecs en fonction de l'état actuel de l'échiquier et de du modèle selectionné.

    Args:
        board (chess.Board): L'échiquier actuel représenté en utilisant la bibliothèque python-chess.
        model (tensorflow.keras.Model): Le modèle d'apprentissage automatique utilisé pour prédire les mouvements.
        move_int_dico (dict): Un dictionnaire qui associe les indices des mouvements aux notations de coups.

    Returns:
        list: Une liste de coups prévus triés par ordre décroissant de probabilité.
    """
    # Obtient la représentation FEN de l'échiquier et le convertir en une matrice adaptée à l'entrée du modèle
    fen_code = board.fen()
    board_matrix = fen_to_matrix(fen_code)
    board_matrix = np.expand_dims(board_matrix, axis=0) # Ajoute une dimension supplémentaire à la matrice (batch size) pour correspondre au format attendu par le modèle
    predictions = model.predict(board_matrix)[0]  # Effectue une prédiction à l'aide du modèle
    sorted_indices = np.argsort(predictions)[::-1]  # Trie les indices des coups prévus par probabilité décroissante
    predicted_moves_sorted = [move_int_dico[idx] for idx in sorted_indices] # Traduit la liste des predictions en coup d'échec
    
    return predicted_moves_sorted

def predict_legal_next_move(board, predicted_moves_sorted):
    """
    Sélectionne le premier coup légal parmi une liste de coups prévus pour un échiquier donné.

    Args:
        board (chess.Board): L'échiquier actuel représenté en utilisant la bibliothèque python-chess.
        predicted_moves_sorted (list): Une liste de coups prévus (notation UCI), triés par probabilité décroissante.

    Returns:
        str or None: Le premier coup légal trouvé sous forme de chaîne (notation UCI),
                     ou None si aucun des coups prédits n'est légal.
    """
    # Parcour les coups prédits et vérifie si le coup est légal dans l'état actuel de l'échiquier
    for move in predicted_moves_sorted:
        chess_move = chess.Move.from_uci(move) # Convertir le coup au format UCI en un objet Move
        if chess_move in board.legal_moves:
            return move  # Retourne le premier coup légal trouvé
    return None # Si aucun coup dans la liste n'est légal, retourner None

def Chess_IA(board, model, move_int_dico):
    """
    Fonction pour jouer un coup au jeu d'échecs en fonction du modèle choisi.
    
    Parameters:
        board : chess.Board
            L'état actuel du plateau d'échecs.
        model : str ou modèle TensorFlow
            Si "random", utilise l'IA aléatoire.
            Sinon, doit être un modèle TensorFlow entraîné.
        move_int_dico : dict
            Dictionnaire de correspondance entre les coups et les indices du modèle.
    
    Returns:
        str
            Le meilleur coup selon le modèle ou un coup aléatoire.
    """
    if model == "random":
        legal_moves = list(board.legal_moves)  # Liste des coups légaux
        return random.choice(legal_moves).uci()  # Choix aléatoire
    
    # Utilisation du modèle TensorFlow pour prédire le meilleur coup
    predicted_moves_list = predict_next_move(board, model, move_int_dico)
    best_legal_predicted_move = predict_legal_next_move(board, predicted_moves_list)
    return best_legal_predicted_move

def play_game(IA_Model_1, IA_Model_2, move_int_dico, print_game=True):
    board = chess.Board()  # Plateau initialisé à la position de départ
    if print_game:
        print("Début de la partie !")
        print(board)

    while not board.is_game_over():  # Tant que la partie n'est pas terminée
        if board.turn:  # Blancs
            move = Chess_IA(board, IA_Model_1, move_int_dico)
        else:  # Noirs
            move = Chess_IA(board, IA_Model_2, move_int_dico)

        # Convertir le coup en objet Move et jouer
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
        else:
            print(f"Erreur : coup illégal {move}")
            break
        
        if print_game:
            print(board, "\n")

    print(f"Résultat : {board.result()}")
    
    return board.result()

def play_n_games(ia1, ia2, n):
    # Initialisation des compteurs pour les résultats
    ia1_wins = 0
    ia2_wins = 0
    draws = 0

    for i in range(n):
        print(f"\nPartie {i+1}:")
        
        # Alterner les couleurs à chaque partie
        if i % 2 == 0:
            # Partie où ia1 joue avec les blancs et ia2 avec les noirs
            result = play_game(ia1, ia2, print_game=False)
        else:
            # Partie où ia2 joue avec les blancs et ia1 avec les noirs
            result = play_game(ia2, ia1, print_game=False)

        # Analyser le résultat de la partie
        if result == "1-0":
            ia1_wins += 1
        elif result == "0-1":
            ia2_wins += 1
        else:
            draws += 1

    # Résultat final
    print("\nRésultats finaux:")
    print(f"Victoires de l'IA n°1 : {ia1_wins}")
    print(f"Victoires de l'IA n°2 : {ia2_wins}")
    print(f"Égalités : {draws}")


#####  Chargement des Données #####
data = pd.read_csv('data/data_cleaned.csv' )

X = data['fen']
y = data['best_move_m1']

##### Traitement finale des données #####

X = X.apply(fen_to_matrix)
X = np.array(X.tolist())

# Encode les mouvements et conversion en catégories
y, move_to_int = encode_moves(y)
y = to_categorical(y, num_classes=len(move_to_int))


# Sauvegarde de move_to_int dans un fichier JSON
with open('Models/move_int_dico.json', 'w') as file:
    json.dump(move_to_int, file)


##### Definition du modele #####
model_1 = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(13, 8, 8)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(move_to_int), activation='softmax')
])

model_1.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model_1.summary()


#### Entrainement #####
model_1.save("Models\Modele_1_TF_25EPOCHS.keras")