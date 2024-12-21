# ML_Chess_2024

Ce projet vise a utiliser un réseau de neurone pour créer une IA capable de jouer aux échecs.

# 1. Collecte des données

**Récupèration un grand nombre de parties d'échecs annotées** . Des bases de données PGN (Portable Game Notation) sont disponibles en ligne sur des sites comme Lichess ou Chess.com.

**BDD utilisé (LitChess) :** 
[132,053,332 chess positions evaluated with Stockfish](https://database.lichess.org/#evals)

--- 

Chaque partie doit inclure :
- Les positions des pièces à chaque coup.

- Le coup joué à partir de cette position.

- l'évaluation de la position donnée par un moteur d'échecs (par exemple Stockfish) pour servir de "vérité terrain".

# 2. Préparer les données
Convertir les parties du format PGN en un format utilisable par le modèle :

## Format de la Base de Données

La base de données contient des informations sur l'état de l'échiquier à chaque étape d'une partie, ainsi que les coups joués. 

Chaque ligne de la base de données correspond à un **coup individuel** joué dans une partie d'échecs, et non à une partie entière. Par exemple, si une partie fait 50 coups, cette partie génèrera 50 lignes dans la base de données.

--- 

Voici un exemple de structure de données après traitement :

| Board_State (8x8 Matrix)                                                                                                                                                                                                                                                                                                                                                                                  | Turn  | Castling | Halfmove Clock | Fullmove Number | Legal Moves           | Best Move      |  
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|----------|----------------|-----------------|-----------------------|----------------|  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | White | KQkq     | 0              | 1               | [e2e4, d2d4, g1f3]   | e2e4          |  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | Black | KQkq     | 0              | 2               | [e7e5, g8f6, b8c6]   | e7e5          |  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | White | KQkq     | 1              | 3               | [d2d4, g1f3, f1c4]   | g1f3          |  

### Détail des colonnes
1. **`Board_State`** : La position actuelle de l'échiquier. Une matrice 8x8, où chaque chiffre représente une pièce :
   - Roi blanc = `6`, reine blanche = `5`, tours = `4`, cavaliers = `2`, fous = `3`, pions = `1`.
   - Les pièces noires sont représentées par les mêmes valeurs en négatif.
   - Les cases vides sont `0`.

2. **`Turn`** : Qui doit jouer le prochain coup (`White` ou `Black`).

3. **`Castling`** : Les droits au roque restants (`KQkq` signifie que les deux camps peuvent encore roquer des deux côtés).

4. **`Halfmove Clock`** : Nombre de demi-coups depuis la dernière prise ou poussée de pion.

5. **`Fullmove Number`** : Numéro du coup dans la partie (commençant à 1).

6. **`Legal Moves`** : La liste des coups légaux à partir de cette position (au format PGN, comme `e2e4`, `g1f3`).

7. **`Best Move`** : Le meilleur coup à jouer, selon l'analyse ou la base de données (variable cible).

### Utilisation dans un réseau de neurones
- **Entrée** : `Board_State`, `Turn`, `Castling`, `Halfmove Clock`, `Fullmove Number`.
- **Sortie** : Une classification parmi les coups possibles (`Legal Moves`) pour prédire le `Best Move`.

# 3. Modèle de réseau de neurones

**Classification multiclasses** :
Entraîne un modèle pour prédire le meilleur coup parmi un ensemble de coups légaux.
Chaque coup est un label distinct.


# 4. Méthodologie

 Commencer par des parties de joueurs haut-classé ainsi que des parties annotées avec des "meilleurs coups" suggérés par des moteurs d'échecs pour entraîner le modèle. 
 
 Cela permettra d'améliorer les performances en entrainant le modèle uniquement sur les meilleurs parties et de gagner en temps d'entrainement.

# 5. Évaluation
Calcule des métriques comme le pourcentage de "coups parfaits" par rapport à un moteur d'échecs ou l'ELO estimé.

Le faire jouer contre d'autres IA / joueurs.

# Outils
TensorFlow ou PyTorch pour implémenter le réseau de neurones.

# Bibliothèques
python-chess pour manipuler les parties et générer les positions.
