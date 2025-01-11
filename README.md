# ML_Chess_2024

Ce projet vise à utiliser un réseau de neurone pour créer une IA capable de jouer aux échecs de manière pertinente et qui se rapproche d'un niveau avancé

## Structure du répertoire GIT 

```bash
C:.
│   .gitignore
│   environnementMargaux.yml
│   README.md
│   requirement.txt
│   Test_Djib.ipynb
│   Test_Theo copy.ipynb
│   Test_Theo.ipynb
│
├───data #jeux de données 
│       data_cleaned.csv #jeux de données récupérées à l'issue de la fonction preprocess
│       lichess_db_eval.jsonl # jeux de données de départ
│
├───Fct
│   │   fct_eval.py #Fonction d'évaluation du modèle
│   │   fct_preprocess.py #Fonctions de récupération du data clean
│   │   __init__
│   │
│   └───__pycache__
│           fct_eval.cpython-310.pyc 
│           fct_eval.cpython-312.pyc
│           fct_preprocess.cpython-310.pyc
│           fct_preprocess.cpython-312.pyc
│
├───Models
│       Modele_1_TF_25EPOCHS.keras 
│       Modele_2_TF_25EPOCHS.keras
│       Modele_3_TF_25EPOCHS.keras
│       move_int_dico.json
│
└───src
        training_chess.py
```

## Prérequis
Charger un evironnement Anaconda à partir du fichier environnementMargaux.yml
Détail des packages :
```bash
channels:
  - conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
  - https://repo.anaconda.com/pkgs/msys2
dependencies:
  - _tflow_select=2.3.0=mkl
  - abseil-cpp=20211102.0=hd77b12b_0
  - absl-py=2.1.0=py310haa95532_0
  - aiohappyeyeballs=2.4.4=py310haa95532_0
  - aiohttp=3.11.10=py310h827c3e9_0
  - aiosignal=1.2.0=pyhd3eb1b0_0
  - asttokens=3.0.0=pyhd8ed1ab_1
  - astunparse=1.6.3=py_0
  - async-timeout=5.0.1=py310haa95532_0
  - attrs=24.3.0=py310haa95532_0
  - blas=1.0=mkl
  - blinker=1.6.2=py310haa95532_0
  - bottleneck=1.4.2=py310hc99e966_0
  - brotli-python=1.0.9=py310h5da7b33_9
  - bzip2=1.0.8=h2bbff1b_6
  - c-ares=1.19.1=h2bbff1b_0
  - ca-certificates=2024.12.14=h56e8100_0
  - cachetools=5.3.3=py310haa95532_0
  - certifi=2024.12.14=pyhd8ed1ab_0
  - cffi=1.17.1=py310h827c3e9_1
  - charset-normalizer=3.3.2=pyhd3eb1b0_0
  - click=8.1.7=py310haa95532_0
  - colorama=0.4.6=py310haa95532_0
  - comm=0.2.2=pyhd8ed1ab_1
  - cryptography=41.0.3=py310h3438e0d_0
  - debugpy=1.8.11=py310h9e98ed7_0
  - decorator=5.1.1=pyhd8ed1ab_1
  - exceptiongroup=1.2.2=pyhd8ed1ab_1
  - executing=2.1.0=pyhd8ed1ab_1
  - flatbuffers=2.0.0=h6c2663c_0
  - frozenlist=1.5.0=py310h827c3e9_0
  - gast=0.4.0=pyhd3eb1b0_0
  - giflib=5.2.2=h7edc060_0
  - google-auth=2.29.0=py310haa95532_0
  - google-auth-oauthlib=0.4.4=pyhd3eb1b0_0
  - google-pasta=0.2.0=pyhd3eb1b0_0
  - grpc-cpp=1.48.2=hf108199_0
  - grpcio=1.48.2=py310hf108199_0
  - h5py=3.12.1=py310h3b2c811_0
  - hdf5=1.12.1=h51c971a_3
  - icc_rt=2022.1.0=h6049295_2
  - icu=58.2=ha925a31_3
  - idna=3.7=py310haa95532_0
  - importlib-metadata=8.5.0=pyha770c72_1
  - intel-openmp=2023.1.0=h59b6b97_46320
  - ipykernel=6.29.5=pyh4bbf305_0
  - ipython=8.31.0=pyh7428d3b_0
  - jedi=0.19.2=pyhd8ed1ab_1
  - jpeg=9e=h827c3e9_3
  - jupyter_client=8.6.3=pyhd8ed1ab_1
  - jupyter_core=5.7.2=py310h5588dad_0
  - keras=2.10.0=py310haa95532_0
  - keras-preprocessing=1.1.2=pyhd3eb1b0_0
  - krb5=1.21.3=h21635db_0
  - libcurl=8.9.1=h0416ee5_0
  - libffi=3.4.4=hd77b12b_1
  - libpng=1.6.39=h8cc25b3_0
  - libprotobuf=3.20.3=h23ce68f_0
  - libsodium=1.0.20=hc70643c_0
  - libssh2=1.10.0=hcd4344a_2
  - markdown=3.4.1=py310haa95532_0
  - markupsafe=2.1.3=py310h2bbff1b_0
  - matplotlib-inline=0.1.7=pyhd8ed1ab_1
  - mkl=2023.1.0=h6b88ed4_46358
  - mkl-service=2.4.0=py310h2bbff1b_1
  - mkl_fft=1.3.11=py310h827c3e9_0
  - mkl_random=1.2.8=py310hc64d2fc_0
  - multidict=6.1.0=py310h827c3e9_0
  - nest-asyncio=1.6.0=pyhd8ed1ab_1
  - numexpr=2.10.1=py310h4cd664f_0
  - numpy=1.26.4=py310h055cbcc_0
  - numpy-base=1.26.4=py310h65a83cf_0
  - oauthlib=3.2.2=py310haa95532_0
  - openssl=1.1.1w=hcfcfb64_0
  - opt_einsum=3.3.0=pyhd3eb1b0_1
  - packaging=24.2=py310haa95532_0
  - pandas=2.2.3=py310h5da7b33_0
  - parso=0.8.4=pyhd8ed1ab_1
  - pickleshare=0.7.5=pyhd8ed1ab_1004
  - pip=24.2=py310haa95532_0
  - platformdirs=4.3.6=pyhd8ed1ab_1
  - prompt-toolkit=3.0.48=pyha770c72_1
  - propcache=0.2.0=py310h827c3e9_0
  - protobuf=3.20.3=py310hd77b12b_0
  - psutil=6.1.1=py310ha8f682b_0
  - pure_eval=0.2.3=pyhd8ed1ab_1
  - pyasn1=0.4.8=pyhd3eb1b0_0
  - pyasn1-modules=0.2.8=py_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pygments=2.19.1=pyhd8ed1ab_0
  - pyjwt=2.10.1=py310haa95532_0
  - pyopenssl=23.2.0=py310haa95532_0
  - pysocks=1.7.1=py310haa95532_0
  - python=3.10.13=h966fe2a_0
  - python-dateutil=2.9.0post0=py310haa95532_2
  - python-flatbuffers=24.3.25=py310haa95532_0
  - python-tzdata=2023.3=pyhd3eb1b0_0
  - python_abi=3.10=2_cp310
  - pytz=2024.1=py310haa95532_0
  - pywin32=307=py310h9e98ed7_3
  - pyzmq=26.2.0=py310h656833d_3
  - re2=2022.04.01=hd77b12b_0
  - requests=2.32.3=py310haa95532_1
  - requests-oauthlib=2.0.0=py310haa95532_0
  - rsa=4.7.2=pyhd3eb1b0_1
  - scipy=1.14.1=py310h8640f81_0
  - setuptools=75.1.0=py310haa95532_0
  - six=1.16.0=pyhd3eb1b0_1
  - snappy=1.2.1=hcdb6601_0
  - sqlite=3.45.3=h2bbff1b_0
  - stack_data=0.6.3=pyhd8ed1ab_1
  - tbb=2021.8.0=h59b6b97_0
  - tensorboard=2.10.0=py310haa95532_0
  - tensorboard-data-server=0.6.1=py310haa95532_0
  - tensorboard-plugin-wit=1.8.1=py310haa95532_0
  - tensorflow=2.10.0=mkl_py310hd99672f_0
  - tensorflow-base=2.10.0=mkl_py310h6a7f48e_0
  - tensorflow-estimator=2.10.0=py310haa95532_0
  - termcolor=2.1.0=py310haa95532_0
  - tk=8.6.14=h0416ee5_0
  - tornado=6.4.2=py310ha8f682b_0
  - traitlets=5.14.3=pyhd8ed1ab_1
  - typing-extensions=4.12.2=py310haa95532_0
  - typing_extensions=4.12.2=py310haa95532_0
  - tzdata=2024b=h04d1e81_0
  - ucrt=10.0.22621.0=h57928b3_1
  - urllib3=2.2.3=py310haa95532_0
  - vc=14.40=haa95532_2
  - vc14_runtime=14.42.34433=he29a5d6_23
  - vs2015_runtime=14.42.34433=hdffcdeb_23
  - wcwidth=0.2.13=pyhd8ed1ab_1
  - werkzeug=3.0.6=py310haa95532_0
  - wheel=0.44.0=py310haa95532_0
  - win_inet_pton=1.1.0=py310haa95532_0
  - wrapt=1.14.1=py310h2bbff1b_0
  - xz=5.4.6=h8cc25b3_1
  - yarl=1.18.0=py310h827c3e9_0
  - zeromq=4.3.5=ha9f60a1_7
  - zipp=3.21.0=pyhd8ed1ab_1
  - zlib=1.2.13=h8cc25b3_1
  - pip:
      - chess==1.11.1
      - json5==0.9.6
```

## Lien du repository
```bash
git clone https://github.com/Tusked34/ML_Chess_2024
```

## Utilisation du repository

1. Prétraiter les données :
```bash

```
2. Entraîner le modèle :
```bash

```

3. Évaluer les performances du modèle :
```bash

```

4. Jouer contre l'IA
```bash

```


# Détail du projet

## 1. Collecte des données

**Récupèration un grand nombre de parties d'échecs annotées** . Des bases de données PGN (Portable Game Notation) sont disponibles en ligne sur des sites comme Lichess ou Chess.com.

**BDD utilisé (LitChess) :** 
[132,053,332 chess positions evaluated with Stockfish](https://database.lichess.org/#evals)

--- 

Chaque partie doit inclure :
- Les positions des pièces à chaque coup.

- Le coup joué à partir de cette position.

- l'évaluation de la position donnée par un moteur d'échecs (par exemple Stockfish) pour servir de "vérité terrain".

## 2. Préparer les données
Convertir les parties du format PGN en un format utilisable par le modèle :

### Format de la Base de Données

La base de données contient des informations sur l'état de l'échiquier à chaque étape d'une partie, ainsi que les coups joués. 

Chaque ligne de la base de données correspond à un **coup individuel** joué dans une partie d'échecs, et non à une partie entière. Par exemple, si une partie fait 50 coups, cette partie génèrera 50 lignes dans la base de données.

--- 

Voici un exemple de structure de données après traitement :

| Board_State (8x8 Matrix)                                                                                                                                                                                                                                                                                                                                                                                  | Turn  | Castling | Halfmove Clock | Fullmove Number | Legal Moves           | Best Move      |  
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|----------|----------------|-----------------|-----------------------|----------------|  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | White | KQkq     | 0              | 1               | [e2e4, d2d4, g1f3]   | e2e4          |  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | Black | KQkq     | 0              | 2               | [e7e5, g8f6, b8c6]   | e7e5          |  
| [[-4, -2, -3, -6, -5, -3, -2, -4], [-1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1], [4, 2, 3, 6, 5, 3, 2, 4]]                                                                                                                              | White | KQkq     | 1              | 3               | [d2d4, g1f3, f1c4]   | g1f3          |  

#### Détail des colonnes
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

#### Utilisation dans un réseau de neurones
- **Entrée** : `Board_State`, `Turn`, `Castling`, `Halfmove Clock`, `Fullmove Number`.
- **Sortie** : Une classification parmi les coups possibles (`Legal Moves`) pour prédire le `Best Move`.

## 3. Modèle de réseau de neurones

**Classification multiclasses** :
Entraîne un modèle pour prédire le meilleur coup parmi un ensemble de coups légaux.
Chaque coup est un label distinct.


## 4. Méthodologie

 Commencer par des parties de joueurs haut-classé ainsi que des parties annotées avec des "meilleurs coups" suggérés par des moteurs d'échecs pour entraîner le modèle. 
 
 Cela permettra d'améliorer les performances en entrainant le modèle uniquement sur les meilleurs parties et de gagner en temps d'entrainement.

## 5. Évaluation
Calcule des métriques comme le pourcentage de "coups parfaits" par rapport à un moteur d'échecs ou l'ELO estimé.

Le faire jouer contre d'autres IA / joueurs.

### Outils
TensorFlow ou PyTorch pour implémenter le réseau de neurones.

### Bibliothèques
python-chess pour manipuler les parties et générer les positions.



