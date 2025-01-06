import chess
import random
from Fct.fct_preprocess import *


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