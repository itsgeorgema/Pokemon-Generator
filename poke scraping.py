import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
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

print(df[900:])