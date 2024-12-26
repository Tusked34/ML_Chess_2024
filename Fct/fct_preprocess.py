
import pandas as pd
import json
import chess
import numpy as np

def extract_data(file_path, max_rows=50):
    data = []
    
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break

            position = json.loads(line)
            fen = position.get("fen")

            for eval_data in position.get("evals", []):
                knodes = eval_data.get("knodes")
                depth = eval_data.get("depth")
                
                for pv in eval_data.get("pvs", []):
                    cp = pv.get("cp")
                    mate = pv.get("mate")
                    line = pv.get("line")

                    data.append({
                        "fen": fen,
                        "knodes": knodes,
                        "depth": depth,
                        "cp": cp,
                        "mate": mate,
                        "line": line
                    })

    return pd.DataFrame(data)


def extract_next_move(df) :
    # Extraction du premier groupe de mots de la colonne 'line'
    df["best_move_m1"] = df["line"].str.extract(r'(\S+)', expand=False)
    return(df)


def most_popular_predict_V1(df):
    most_popular_moves = (
        df.groupby('fen')['best_move_m1']
        .agg(lambda x: x.value_counts().idxmax())  # Trouver le coup le plus fréquent
        .rename("best_move_m2")
    )

    # Ajouter cette information dans le DataFrame original
    df = df.merge(most_popular_moves, on='fen', how='left')
    return df


def filter_best_predict(df):
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
    df = df.drop(columns=['line','mate','cp','depth','knodes'])
    return df


def fen_to_matrix(fen: str):
    """
    Utilité: 
        Convertit une position d'échecs en FEN en une matrice 8x8x12.
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
    Utilité :
        Cette fonction prend une liste de coups d'échecs (sous forme de chaînes) et encode chaque coup unique
        sous forme d'un entier unique. Elle retourne également un dictionnaire qui associe chaque coup à son
        entier correspondant.

    Parameters:
        moves (list of str): A list of chess moves as strings (e.g., ['e2e4', 'd2d4']).

    Returns:
        tuple:
            - A list of integers representing the encoded moves (e.g., [0, 1, 0]).
            - A dictionary mapping each unique move to its corresponding integer (e.g., {'e2e4': 0, 'd2d4': 1}).

    """
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int