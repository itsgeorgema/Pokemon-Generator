import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.utils import save_image

metadata=pd.read_csv("pokemon_data_pokeapi.csv")
types=sorted(list(set(metadata['Type1'].dropna().tolist() + metadata['Type2'].dropna().tolist())))
max_height=metadata['Height (m)'].max()
max_weight=metadata['Weight (kg)'].max()
max_gen=metadata['Generation'].max()

def one_hot_type(t):
    vector=[0]*len(types)
    if t in types:
        vector[types.index(t)]=1
    return vector

#GENERATOR WITHOUT DISCRIMINATOR TO CREATE FAKE IMAGES
class Generator(nn.Module):
    def __init__(self, z_dim, condition_dim, img_channels=3, feature_g=64):
        super().__init__()
        self.fc =nn.Linear(z_dim + condition_dim, feature_g * 8 * 4 * 4)
        self.gen=nn.Sequential(nn.BatchNorm2d(feature_g* 8),
              nn.ReLU(True), nn.ConvTranspose2d(feature_g*8, feature_g*4,4,2,1), nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),nn.ConvTranspose2d(feature_g*4, feature_g*2,4,2,1),nn.BatchNorm2d(feature_g*2),nn.ReLU(True),
            nn.ConvTranspose2d(feature_g* 2, feature_g,4,2,1),
            nn.BatchNorm2d(feature_g),nn.ReLU(True),nn.ConvTranspose2d(feature_g, img_channels,4,2,1), nn.Tanh())

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)  # assumes feature_g = 64 → 512=64*8
        return self.gen(x)



def generate_image(generator, condition,z_dim=100):
    z= torch.randn(1,z_dim)
    with torch.no_grad():
        fake_img=generator(z,condition)
    return fake_img[0].permute(1, 2, 0).cpu().numpy()*.5+.5
    def generate_and_save_image(type1, type2, height, weight, generation, legendary, output_path="static/generated/image.png"):
    z_dim = 100
    condition_dim = len(one_hot_type("Fire")) * 2 + 4
    generator = Generator(z_dim=z_dim, condition_dim=condition_dim)
    checkpoint = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    cond_vector = one_hot_type(type1) + one_hot_type(type2) + [
        height / max_height,
        weight / max_weight,
        generation / max_gen,
        1.0 if legendary else 0.0
    ]
    cond_tensor = torch.tensor(cond_vector).unsqueeze(0).float()
    z = torch.randn(1, z_dim)
    with torch.no_grad():
        fake_img = generator(z, cond_tensor)
    save_image(fake_img, output_path)
    return output_path
#PROMPT USEER TO CREATE A POKEMON BY ASKING FOR ATTRIBUTES
# def get_user_input():
#     print("Pokemon Creator:")
#     first_type=input("Enter Type 1 (e.g., Fire): ").capitalize()
#     second_type=input("Enter Type 2 (or press Enter for None): ").capitalize()
#     if not second_type:
#         second_type=None
#     height=float(input("Height in meters (e.g., 1.2): "))
#     weight=float(input("Weight in kg (e.g., 35): "))
#     gen=int(input("Generation (1-8): "))
#     legendary_status=input("Legendary? (y/n): ").lower()=="y"
#     cond_vector=one_hot_type(first_type)+one_hot_type(second_type)+[height/max_height,weight/max_weight,gen/max_gen,
#     1.0 if legendary_status else 0.0]
#     return torch.tensor(cond_vector).unsqueeze(0).float()

# def main():
#     z_dim=100
#     condition_dim=len(one_hot_type("Fire"))*2+4
#     generator=Generator(z_dim=z_dim,condition_dim=condition_dim)
#     checkpoint=torch.load("checkpoint.pth",location=torch.device('cpu'))
#     generator.load_state_dict(checkpoint["generator_state_dict"])
#     generator.eval()
#     plt.imshow(generate_image(generator, get_user_input()))
#     plt.title("Generated Pokémon")
#     plt.axis("off")
#     plt.show()



# if __name__=="__main__":
#     main()
