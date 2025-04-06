#code goes here

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as nan
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import random

df1 = pd.read_csv("pokemon_data_pokeapi.csv")
df1 = df1.drop(columns = ['Pokedex Number', 'Classification','Height (m)', 'Weight (kg)', 'Generation', 'Legendary Status'])
df2 = pd.read_csv("Pokemon_stats.csv")
df2 = df2.drop(columns=['Type 1', 'Type 2', '#', 'Legendary', 'Generation'])

df1['ability1'] = [i.split(", ")[0] for i in df1['Abilities']]
df = pd.merge(df1, df2, on='Name', how='inner')
print(df)


type_encoder = LabelEncoder()
df['Type1_encoded'] = type_encoder.fit_transform(df['Type1'])
df['Type2_encoded'] = type_encoder.fit_transform(df['Type2'])

rf_hp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_hp.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['HP'])

rf_def = RandomForestRegressor(n_estimators=100, random_state=42)
rf_def.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['Defense'])

rf_atk = RandomForestRegressor(n_estimators=100, random_state=42)
rf_atk.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['Attack'])

rf_satk = RandomForestRegressor(n_estimators=100, random_state=42)
rf_satk.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['Sp. Atk'])

rf_sdef = RandomForestRegressor(n_estimators=100, random_state=42)
rf_sdef.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['Sp. Def'])

rf_spd = RandomForestRegressor(n_estimators=100, random_state=42)
rf_spd.fit(df[['Type1_encoded', 'Type2_encoded']].fillna(-1), df['Speed'])

def random_ability():
    all_abilities = df['Abilities'].str.split(',', expand=True).stack().reset_index(drop=True)
    ability_counts = Counter(all_abilities)
    common_abilities = {ability: count for ability, count in ability_counts.items() if count >= 2}
    abilities_list = list(common_abilities.keys())
    weights = list(common_abilities.values())
    return random.choices(abilities_list, weights=weights, k=1)[0]

def predict_smoothness(type1, type2=None):
    type1_encoded = type_encoder.transform([type1])[0] 
    
    if type2 is None:
        predicted_hp = rf_hp.predict([[type1_encoded, -1]])[0]
        predicted_atk = rf_atk.predict([[type1_encoded, -1]])[0]
        predicted_def = rf_def.predict([[type1_encoded, -1]])[0]
        predicted_satk = rf_satk.predict([[type1_encoded, -1]])[0]
        predicted_sdef = rf_sdef.predict([[type1_encoded, -1]])[0]
        predicted_spd = rf_spd.predict([[type1_encoded, -1]])[0]
    else:
        type2_encoded = type_encoder.transform([type2])[0]
        predicted_hp = rf_hp.predict([[type1_encoded, type2_encoded]])[0]
        predicted_atk = rf_atk.predict([[type1_encoded, type2_encoded]])[0]
        predicted_def = rf_def.predict([[type1_encoded, type2_encoded]])[0]
        predicted_satk = rf_satk.predict([[type1_encoded, type2_encoded]])[0]
        predicted_sdef = rf_sdef.predict([[type1_encoded, type2_encoded]])[0]
        predicted_spd = rf_spd.predict([[type1_encoded, type2_encoded]])[0]
    
    hp = round(predicted_hp)
    atk = round(predicted_atk)
    defen = round(predicted_def)
    satk = round(predicted_satk)
    sdef = round(predicted_sdef)
    spd = round(predicted_spd)
    iv = hp + atk + defen + satk + sdef + spd
    
    return (f"Type 1: {type1} and Type 2: {type2 if type2 else 'None'}:\n"
            f"  IV: {iv}\n"
            f"  HP: {hp}\n"
            f"  Attack: {atk}\n"
            f"  Defense: {defen}\n"
            f"  Special Attack: {satk}\n"
            f"  Special Defense: {sdef}\n"
            f"  Speed: {spd}\n"
            f"  Random Ability: {random_ability()}\n")

def get_user_input():
    type1 = input("Enter Type 1: ").strip().capitalize() 
    type2_input = input("Enter Type 2 (or leave blank if missing): ").strip().capitalize() 
    type2 = type2_input if type2_input else None
    
    result = predict_smoothness(type1, type2)
    print(result)

get_user_input()
