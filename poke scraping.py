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