import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df1 = pd.read_csv("pokemon_data_pokeapi.csv")
df1 = df1.drop(columns = ['Pokedex Number', 'Classification','Height (m)', 'Weight (kg)', 'Generation', 'Legendary Status'])
df2 = pd.read_csv("Pokemon_stats.csv")
df2 = df2.drop(columns=['Type 1', 'Type 2', '#', 'Legendary', 'Generation'])

df1['ability1'] = [i.split(", ")[0] for i in df1['Abilities']]
df = pd.merge(df1, df2, on='Name', how='inner')
print(df)
#print(first_ability)


#sns.stripplot(x='Type2', y='Type1', data=df, jitter=True, palette='Set2')
#plt.title('Distribution of Type2 by Type1')
#plt.xticks(rotation=45)
#plt.show()


#encoding
type_encoder, ability_encoder = LabelEncoder(), LabelEncoder()

# Encode 'Source', 'Rock_1', and 'Rock_2' columns
df['type1_encoded'] = type_encoder.fit_transform(df['Type1'])
df['ability_encoded'] = ability_encoder.fit_transform(df['ability1'])

df['Type2'] = df['Type2'].fillna('None')
df['type2_encoded'] = type_encoder.fit_transform(df['Type2'])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df[['type1_encoded', 'type2_encoded']], df['ability_encoded'])

rf_type2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_type2.fit(df[['type1_encoded', 'ability_encoded']], df['type2_encoded'])

def predict_type2_ability(type1, type2=None):
    type1_encoded = type_encoder.transform([type1])[0]
    
    if type2 is None:
        predicted_type_encoded = rf_type2.predict([[type1_encoded, 0]])
        predicted_type_name = type_encoder.inverse_transform(predicted_type_encoded)[0]
    else:
        predicted_type_encoded = type_encoder.transform([type2])[0]
        predicted_type_name = type2
    
    predicted_ability_encoded = rf.predict([[type1_encoded, predicted_type_encoded]])
    predicted_ability_name = ability_encoder.inverse_transform(predicted_ability_encoded)[0]
    
    return predicted_type_name, predicted_ability_name

def get_user_input():
    type1in = input("Enter Type 1: ").strip().capitalize() 
    type2in = input("Enter Type 2 (or press enter if none): ").strip().capitalize()
    
    type2 = type2in if type2in else None
    
    return type1in, type2
    

type1, type2 = get_user_input()

predicted_type2, predicted_ability = predict_type2_ability(type1, type2)

if type2 is None:
    print(f"Type 1: {type1}, predicted Type 2: {predicted_type2} and predicted ability is: {predicted_ability}")
else:
    print(f"Type 1: {type1} and Type 2 : {type2}, predicted ability is: {predicted_ability}")