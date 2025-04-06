#code goes here

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
from flask import Flask, render_template, request, jsonify
import random
import requests
from bs4 import BeautifulSoup
import re
from CreateImage import generate_and_save_image

def get_df():
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
    return df

df = get_df()

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
    all_abilities = [ability for sublist in df['Abilities'] for ability in sublist]
    ability_counts = Counter(all_abilities)
    common_abilities = {ability: count for ability, count in ability_counts.items() if count >= 2}
    abilities_list = list(common_abilities.keys())
    weights = list(common_abilities.values())
    return random.choices(abilities_list, weights=weights, k=1)[0]

def add_noise_to_stats(stats, noise_level=0.1):
    noisy_stats = {}
    for stat_name, stat_value in stats.items():
        noise = random.uniform(-noise_level, noise_level)
        noisy_stats[stat_name] = stat_value * (1 + noise)
    return noisy_stats

def scale_stats(stats, min_total=120, max_total=510, max_stat=255):
    total_stats = sum(stats.values())

    if total_stats > max_total:
        scaling_factor = max_total / total_stats
        stats = {key: min(value * scaling_factor, max_stat) for key, value in stats.items()}
    elif total_stats < min_total:
        scaling_factor = min_total / total_stats
        stats = {key: min(value * scaling_factor, max_stat) for key, value in stats.items()}

    return stats

def predict_smoothness(type1, type2=None):
    type1_encoded = type_encoder.transform([type1])[0]
    
    if type2 is None:
        pred_hp = rf_hp.predict([[type1_encoded, -1]])[0]
        pred_atk = rf_atk.predict([[type1_encoded, -1]])[0]
        pred_def = rf_def.predict([[type1_encoded, -1]])[0]
        pred_satk = rf_satk.predict([[type1_encoded, -1]])[0]
        pred_sdef = rf_sdef.predict([[type1_encoded, -1]])[0]
        pred_spd = rf_spd.predict([[type1_encoded, -1]])[0]
    else:
        type2_encoded = type_encoder.transform([type2])[0]
        pred_hp = rf_hp.predict([[type1_encoded, type2_encoded]])[0]
        pred_atk = rf_atk.predict([[type1_encoded, type2_encoded]])[0]
        pred_def = rf_def.predict([[type1_encoded, type2_encoded]])[0]
        pred_satk = rf_satk.predict([[type1_encoded, type2_encoded]])[0]
        pred_sdef = rf_sdef.predict([[type1_encoded, type2_encoded]])[0]
        pred_spd = rf_spd.predict([[type1_encoded, type2_encoded]])[0]

    stats = {
        "HP": pred_hp,
        "Attack": pred_atk,
        "Defense": pred_def,
        "Sp. Atk": pred_satk,
        "Sp. Def": pred_sdef,
        "Speed": pred_spd
    }

    noisy_stats = add_noise_to_stats(stats, noise_level=0.1)
    final_stats = scale_stats(noisy_stats, min_total=120, max_total=510, max_stat=255)
    final_stats = {key: round(value) for key, value in final_stats.items()}
    
    return {
        "Type1": type1,
        "Type2": type2,
        **final_stats,
        "Total": round(sum(final_stats.values()), 2),
        "ability": random_ability()
    }
    """
    hp = round(predicted_hp)
    atk = round(predicted_atk)
    defen = round(predicted_def)
    satk = round(predicted_satk)
    sdef = round(predicted_sdef)
    spd = round(predicted_spd)



    iv = hp + atk + defen + satk + sdef + spd
    
    return {"type1":type1, "type2":type2, "iv":iv, "hp":hp, "atk":atk, "defen":defen, "satk":satk, "sdef":sdef, "spd":spd,"ability":random_ability()}
"""
def get_user_input():
    type1 = input("Enter Type 1: ").strip().capitalize() 
    type2_input = input("Enter Type 2 (or leave blank if missing): ").strip().capitalize() 
    type2 = type2_input if type2_input else None
    
    result = predict_smoothness(type1, type2)
    print(result)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        #print("Received data:", data)

        required_keys = ['type1', 'type2', 'height', 'weight', 'generation', 'legendary']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        type1 = data["type1"]
        type2 = data["type2"]
        height = float(data["height"])
        weight = float(data["weight"])
        generation = int(data["generation"])
        legendary = bool(data["legendary"])

        stats = predict_smoothness(type1, type2)
        image_path = generate_and_save_image(type1, type2, height, weight, generation, legendary)

        return jsonify({
            "type1": type1,
            "type2": type2,
            "ability": stats["ability"],
            "stats": {k: v for k, v in stats.items() if k != "ability"},
            "image_url": "/" + image_path
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 400

#get_user_input()
if __name__ == '__main__':
    app.run(debug=True)
