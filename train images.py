import os
import pandas as pd

metadata = pd.read_csv("pokemon_data_pokeapi.csv")

image_root = "images" 

merged_data = []

for x, row in metadata.iterrows():
    name = row['Name']
    folder_path = os.path.join(image_root, name)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg')):
                image_path = os.path.join(folder_path, file)
                merged_row = row.to_dict()
                merged_row['image_path'] = image_path
                merged_data.append(merged_row)

merged_df = pd.DataFrame(merged_data)


print(merged_df)

"""import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread(f'{merged_df.get("image_path").loc[2343]}')
plt.imshow(image)
plt.show()"""