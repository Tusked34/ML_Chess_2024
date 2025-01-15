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
    """
    Simule une partie d'échecs entre deux modèles d'intelligence artificielle (IA).

    Args:
        IA_Model_1 (object): Modèle d'IA utilisé pour jouer les coups des Blancs. 
                             Il doit être compatible avec la fonction `Chess_IA`.
        IA_Model_2 (object): Modèle d'IA utilisé pour jouer les coups des Noirs. 
                             Il doit également être compatible avec la fonction `Chess_IA`.
        move_int_dico (dict): Dictionnaire permettant de traduire ou mapper les mouvements 
                              entre un format interne et un format UCI (Universal Chess Interface).
        print_game (bool, optional): Indique si l'état de la partie doit être affiché après chaque coup. 
                                     Par défaut, cette valeur est définie sur `True`.

    Returns:
        str: Le résultat de la partie selon la notation standard de la bibliothèque `python-chess` :
             - "1-0" : Les Blancs gagnent.
             - "0-1" : Les Noirs gagnent.
             - "1/2-1/2" : Partie nulle.
    """
    board = chess.Board()  # Plateau initialisé à la position de départ
    if print_game:
        print("Début de la partie !")
        print(board)

    while not board.is_game_over():  # Tant que la partie n'est pas terminée
        if board.turn:  # Blancs
            move = Chess_IA(board, IA_Model_1, move_int_dico)
            print(f"Blancs jouent : {move}")
        else:  # Noirs
            move = Chess_IA(board, IA_Model_2, move_int_dico)
            print(f"Noirs jouent : {move}")

        # Ajouter la promotion si nécessaire
        move_obj = chess.Move.from_uci(move)
        if move_obj.promotion is None and board.is_legal(move_obj) and board.is_pseudo_legal(move_obj):
            # Si le coup est une promotion, ajouter la promotion par défaut à la reine
            if board.piece_at(move_obj.from_square).piece_type == chess.PAWN and chess.square_rank(move_obj.to_square) in [0, 7]:
                move_obj.promotion = chess.QUEEN  # Promotion par défaut en reine

        if move_obj in board.legal_moves:
            board.push(move_obj)
        else:
            print(f"Erreur : coup illégal {move}")
            break
        
        if print_game:
            print(board, "\n")

    print(f"Résultat : {board.result()}")
    return board.result()


def play_multiple_games(IA_Model_1, IA_Model_2, move_int_dico, num_games):
    """
    Joue N parties d'échecs entre deux IA et retourne les résultats cumulés.

    Parameters:
    - IA_Model_1: Modèle IA pour le joueur 1.
    - IA_Model_2: Modèle IA pour le joueur 2.
    - move_int_dico: Dictionnaire pour les coups.
    - num_games: Nombre de parties à jouer.

    Returns:
    - Un dictionnaire contenant les résultats cumulés.
    """
    results = {"Joueur 1": 0, "Joueur 2": 0, "Draws": 0}

    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(IA_Model_1, IA_Model_2, move_int_dico, print_game=False)
        else:
            # Inverser les modèles pour alterner les couleurs
            result = play_game(IA_Model_2, IA_Model_1, move_int_dico, print_game=False)

        print(f"Partie {i + 1} : Résultat = {result}")

        if result == "1-0":
            if i % 2 == 0:
                results["Joueur 1"] += 1
            else:
                results["Joueur 2"] += 1
        elif result == "0-1":
            if i % 2 == 0:
                results["Joueur 2"] += 1
            else:
                results["Joueur 1"] += 1
        else:
            results["Draws"] += 1

    print("\nRésultats finaux :")
    print(f"Joueur 1 gagne : {results['Joueur 1']} parties")
    print(f"Joueur 2 gagne : {results['Joueur 2']} parties")
    print(f"Parties nulles : {results['Draws']} parties")

    return results


def play_human_vs_ia(IA_Model, move_int_dico, human_color='white', print_game=True):
    """
    Fonction pour jouer une partie contre une IA.
    
    Args:
        IA_Model: Modèle IA pour l'adversaire.
        move_int_dico: Dictionnaire pour convertir les coups.
        human_color: 'white' ou 'black' pour choisir la couleur du joueur humain.
        print_game: bool, afficher ou non le plateau après chaque coup.

    Returns:
        str: Résultat de la partie ("1-0", "0-1", "1/2-1/2").
    """
    board = chess.Board()  # Initialisation du plateau
    is_human_white = human_color.lower() == 'white'

    if print_game:
        print("Début de la partie !")
        print(board)

    while not board.is_game_over():
        if board.turn:  # Tour des blancs
            if is_human_white:  # Humain joue les blancs
                move = str(input("Votre coup (notation UCI, ex : e2e4) : "))
            else:  # IA joue les blancs
                move = Chess_IA(board, IA_Model, move_int_dico)
                print(f"\nl'IA joue : {move}\n")

        else:  # Tour des noirs
            if not is_human_white:  # Humain joue les noirs
                move = str(input("Votre coup (notation UCI, ex : e7e5) : "))
            else:  # IA joue les noirs
                move = Chess_IA(board, IA_Model, move_int_dico)
                print(f"\nl'IA joue : {move}\n")

        # Convertir le coup en objet Move et jouer
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
        else:
            print(f"Erreur : coup illégal {move}")
            continue

        if print_game:
            print(board, "\n")

    print(f"Résultat : {board.result()}")

    return board.result()