#####################
### Bibliothèques ###
#####################
import pandas as pd
import json
import chess
import numpy as np

from Fct.fct_preprocess import *

#################################
### extractions données bruts ###
#################################
file_path = 'data/lichess_db_eval.jsonl'
max_rows = 1000000
data_raw = extract_data(file_path, max_rows)

print("extractions données OK \n")

###################################
#### Transformation des données ###
###################################
data_V1 = extract_next_move(data_raw)
data_V2 = most_popular_predict_V1(data_V1)
data_V3 = filter_best_predict(data_V2)
data_cleaned = drop_usuless_columns(data_V3)

print("Transformation des données OK \n")

##############################
### Sauvegarde des données ###
##############################
file_name = f"data/data_cleaned_{max_rows}_rows.csv"
data_cleaned.to_csv(file_name)

print(f"Sauvegarde des données dans {file_name} OK \n")