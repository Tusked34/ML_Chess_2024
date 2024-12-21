Pour représenter les données après traitement, chaque ligne de la base correspond à une position spécifique d'une partie et le meilleur coup à jouer. Voici à quoi cela pourrait ressembler sous une forme tabulaire ou DataFrame, après le traitement des parties d'échecs :

Board_State (8x8 Matrix)	Turn	Castling	Halfmove Clock	Fullmove Number	Legal Moves	Best Move
[[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]	White	KQkq	0	1	[e2e4, d2d4, g1f3]	e2e4
[[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]	Black	KQkq	0	2	[e7e5, g8f6, b8c6]	e7e5
[[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]	White	KQkq	1	3	[d2d4, g1f3, f1c4]	g1f3
Détail des colonnes
Board_State : La position actuelle de l'échiquier. Une matrice 8x8, où chaque chiffre représente une pièce :

Roi blanc = 6, reine blanche = 5, tours = 4, cavaliers = 2, fous = 3, pions = 1.
Les pièces noires sont représentées par les mêmes valeurs en négatif.
Les cases vides sont 0.
Turn : Qui doit jouer le prochain coup (White ou Black).

Castling : Les droits au roque restants (KQkq signifie que les deux camps peuvent encore roquer des deux côtés).

Halfmove Clock : Nombre de demi-coups depuis la dernière prise ou poussée de pion.

Fullmove Number : Numéro du coup dans la partie (commençant à 1).

Legal Moves : La liste des coups légaux à partir de cette position (au format PGN, comme e2e4, g1f3).

Best Move : Le meilleur coup à jouer, selon l'analyse ou la base de données (variable cible).

Exemple d'utilisation dans un réseau de neurones
Entrée : Board_State, Turn, Castling, Halfmove Clock, Fullmove Number.
Sortie : Une classification parmi les coups possibles (Legal Moves) pour prédire le Best Move.