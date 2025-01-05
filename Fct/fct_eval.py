import chess
import random
from Fct.fct_preprocess import *

def play_game(ia1, ia2, print_game=True):
    board = chess.Board()  # Plateau initialisé à la position de départ
    if print_game:
        print("Début de la partie !")
        print(board)

    while not board.is_game_over():  # Tant que la partie n'est pas terminée
        if board.turn:  # Blancs
            move = ia1(board)
        else:  # Noirs
            move = ia2(board)

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

def random_player(board):
    legal_moves = list(board.legal_moves)  # Liste des coups légaux
    return random.choice(legal_moves).uci()  # Choix aléatoire

def dummy_player(board):
    # Exemple d'IA qui renvoie toujours un coup valide (à adapter selon votre IA)
    legal_moves = list(board.legal_moves)
    return legal_moves[0].uci()  # Prend le premier coup légal (dummy behavior)

def predict_next_move(board, model,move_int_dico):
    # Obtenir le FEN de l'échiquier
    fen_code = board.fen()
    # Convertir le FEN en matrice
    board_matrix = fen_to_matrix(fen_code)
    # Ajouter une dimension batch à la matrice pour correspondre à l'entrée du modèle
    board_matrix = np.expand_dims(board_matrix, axis=0)
    # Effectuer la prédiction
    predictions = model.predict(board_matrix)[0]

    # Crée une nouvelle variable qui contient les indices des prédictions triées par probabilité décroissante
    sorted_indices = np.argsort(predictions)[::-1]  # Trie les indices de predictions de la plus haute à la plus basse

    # Utilise ces indices pour obtenir les coups associés dans l'ordre
    predicted_moves_sorted = [move_int_dico[idx] for idx in sorted_indices]

    return predicted_moves_sorted
