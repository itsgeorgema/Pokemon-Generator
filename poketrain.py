import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("pokemon_data_pokeapi.csv")
df['Type2'] = df['Type2'].fillna('None')

#first_ability = [i.split(", ")[0] for i in df['Abilities']]
#print(first_ability)


#sns.stripplot(x='Type2', y='Type1', data=df, jitter=True, palette='Set2')
#plt.title('Distribution of Type2 by Type1')
#plt.xticks(rotation=45)
#plt.show()



second_encoder = LabelEncoder()
df['Type2_encoded'] = second_encoder.fit_transform(df['Type2'])

first_encoder = LabelEncoder()
df['Type1_encoded'] = first_encoder.fit_transform(df['Type1'])

X = df[['Type1_encoded']]
y = df['Type2_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')


def predict_source(rock_name):
    first_encoded = first_encoder.transform([rock_name])[0]
    
    second_encoded = model.predict([[first_encoded]])[0]
    
    second_predicted = second_encoder.inverse_transform([second_encoded])[0]
    
    return second_predicted

type1 = input("Enter the type 1 to predict its type2: ")
predicted_source = predict_source(type1)
print(f'The predicted type 2 for {type1} is: {predicted_source}')