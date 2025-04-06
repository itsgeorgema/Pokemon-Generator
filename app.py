#code goes here

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
#from flask import Flask, render_template, request, jsonify
import random
import requests
from bs4 import BeautifulSoup
import re

url = "https://www.serebii.net/pokemon/nationalpokedex.shtml"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
wrapper_div = soup.find('div', id='wrapper')
content_div = soup.find('div', id='content')
main_tag = content_div.find('main')
table_tag = main_tag.find('table', class_='dextable', align='center')
            
rows = []

for row in table_tag.find_all('tr')[2:]:
    cols = row.find_all('td')
    if len(cols) >= 11:
        name = cols[3].get_text(strip=True)

        type_tags = cols[4].find_all('a')
        types = [type_tag['href'].split('/')[-1] for type_tag in type_tags]

        type1 =  types[0].capitalize()
        type2 = types[1].capitalize() if len(types) > 1 else None
   
        ability_column = cols[5]
        ability_list = []
        for a_tag in ability_column.find_all('a'):
            ability_name = a_tag.get_text().strip()
            ability_list.append(ability_name)
    
        hp = cols[6].get_text(strip=True)
        att = cols[7].get_text(strip=True)
        def_ = cols[8].get_text(strip=True)
        s_atk = cols[9].get_text(strip=True)
        s_def = cols[10].get_text(strip=True)
        spd = cols[11].get_text(strip=True)
        
        rows.append([name, type1, type2, ability_list, hp, att, def_, s_atk, s_def, spd])

df = pd.DataFrame(rows)
df.columns = ["Name", "Type1", "Type2", "Abilities", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

#df = pd.DataFrame(rows)

"""
df1 = pd.read_csv("pokemon_data_pokeapi.csv")
df1 = df1.drop(columns = ['Pokedex Number', 'Classification','Height (m)', 'Weight (kg)', 'Generation', 'Legendary Status'])
df2 = pd.read_csv("Pokemon_stats.csv")
df2 = df2.drop(columns=['Type 1', 'Type 2', '#', 'Legendary', 'Generation'])

df1['ability1'] = [i.split(", ")[0] for i in df1['Abilities']]
df = pd.merge(df1, df2, on='Name', how='inner')
print(df)
"""


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
    #all_abilities = df['Abilities'].str.split(',', expand=True).stack().reset_index(drop=True)
    all_abilities = [ability for sublist in df['Abilities'] for ability in sublist]
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
    
    return {"type1":type1, "type2":type2, "iv":iv, "hp":hp, "atk":atk, "defen":defen, "satk":satk, "sdef":sdef, "ability":random_ability()}

def get_user_input():
    type1 = input("Enter Type 1: ").strip().capitalize() 
    type2_input = input("Enter Type 2 (or leave blank if missing): ").strip().capitalize() 
    type2 = type2_input if type2_input else None
    
    result = predict_smoothness(type1, type2)
    print(result)
"""
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    type1 = data.get("type1")
    type2 = data.get("type2")

    try:
        stats = predict_smoothness(type1, type2)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "type1": type1,
        "type2": type2,
        "ability": stats["ability"],
        "stats": {k: v for k, v in stats.items() if k != "ability"},
        "image_url": "https://zukan.pokemon.co.jp/zukan-api/up/images/index/e10f25b88cfd78ee822f46d234e4768f.png"
    })

#get_user_input()
if __name__ == '__main__':
    app.run(debug=True)
"""