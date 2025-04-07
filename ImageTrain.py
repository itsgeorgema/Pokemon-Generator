import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#
#MERGE DATA FROM POKEMON INFO CSV WITH MULTIPLE IMAGES FOR EACH POKEMON
metadata=pd.read_csv("pokemon_data_pokeapi.csv")
image_folder_path="images" 
merged_data=[]
for x, row in metadata.iterrows():
    name=row['Name']
    folder_path= os.path.join(image_folder_path, name)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.png','.jpg')):
                image_path=os.path.join(folder_path, file)
                merged_row=row.to_dict()
                merged_row['image_path']=image_path
                merged_data.append(merged_row)
merged_df = pd.DataFrame(merged_data)

#get rid of broken/corrupted image paths
#good_paths=[]
#for path in tqdm(merged_df['image_path']):
#    try:
#        img=Image.open(path)
#        img.verify()
#        valid_paths.append(path)
#    except:
#       print(f"Corrupted image removed: {path}")
#merged_df= merged_df[merged_df['image_path'].isin(good_paths)].reset_index(drop=True)

#CREATE DATASET FIR FOR PYTORCH
class PokemonDataset(Dataset):
    def __init__(self, df, image_size=64):
        self.df=df.reset_index(drop=True)
        self.transform=transforms.Compose([
            transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
        self.types= sorted(list(set(self.df['Type1'].dropna().tolist() + self.df['Type2'].dropna().tolist())))
        self.gen_max=self.df['Generation'].max()
        self.height_max=self.df['Height (m)'].max()
        self.weight_max =self.df['Weight (kg)'].max()

    def one_hot_type(self, t):
        vector=[0]*len(self.types)
        if pd.notna(t) and t in self.types:
            vector[self.types.index(t)]=1
        return vector
    
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        condition=self.one_hot_type(row['Type1'])+self.one_hot_type(row['Type2'])+[row['Height (m)']/self.height_max,
            row['Weight (kg)']/self.weight_max,row['Generation']/self.gen_max,1.0 if row['Legendary Status'] else 0.0]
        try:
            img=Image.open(row['image_path']).convert("RGB")
            img = self.transform(img)
            # Ensure image is 64x64
            if img.shape[1:]!=(64,64):
                print(f"[Fixing Size] image was {img.shape}, resizing to (64, 64)")
                img=F.interpolate(img.unsqueeze(0), size=(64, 64),mode='bilinear',align_corners=False).squeeze(0)
        except Exception as e:
            print(f"[Error loading image: {row['image_path']}] {e}")
            img=torch.zeros(3,64,64)
        return img,torch.tensor(condition).float()

    def __len__(self):
        return len(self.df)

#CREATE GENERATOR
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
    
##CREATE DISCRIMINATOR TO COMPETE AGAINST GENERATOR
class Discriminator(nn.Module):
    def __init__(self, condition_dim,img_channels=3,feature_d=64):
        super().__init__()
        self.condition_fc=nn.Linear(condition_dim, 64 * 64)
        self.disc=nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_d,4,2,1),nn.LeakyReLU(.2,inplace=True),nn.Conv2d(feature_d, feature_d*2,4,2,1),    
            nn.BatchNorm2d(feature_d*2),nn.LeakyReLU(.2, inplace=True),nn.Conv2d(feature_d* 2, feature_d*4,4,2,1),
            nn.BatchNorm2d(feature_d*4),nn.LeakyReLU(.2, inplace=True),nn.Conv2d(feature_d*4,1,8), nn.Sigmoid() )

    def forward(self,img,condition):
        batch_size=img.size(0)
        cond_map= self.condition_fc(condition).view(batch_size,1,64,64)
        if cond_map.shape[2:]!= img.shape[2:]:
            raise ValueError(f"[!] Shape mismatch: img.shape={img.shape}, cond_map.shape={cond_map.shape}")
        x=torch.cat([img, cond_map],dim=1)
        return self.disc(x).view(-1,1)

#TRAIN GENERATIVE ADVERSARIAL NWETWROK
def train_gan(generator,discriminator,dataloader, z_dim, condition_dim, num_epochs=500, start_epoch=0, g_opt=None, d_opt=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    criterion=nn.BCELoss()
    if g_opt is None:
        g_opt=torch.optim.Adam(generator.parameters(),lr=2e-4,betas=(.5,.999))
    if d_opt is None:
        d_opt= torch.optim.Adam(discriminator.parameters(),lr=2e-4,betas=(.5,.999))

    for epoch in range(start_epoch,start_epoch+num_epochs):
        for imgs,conds in dataloader:
            imgs,conds=imgs.to(device),conds.to(device)
            bs =imgs.size(0)
            real_labels=torch.ones(bs, 1).to(device)
            fake_labels=torch.zeros(bs,1).to(device)

            # TRAIN DSICIMRINATOR
            z=torch.randn(bs, z_dim).to(device)
            fake_imgs=generator(z, conds)
            d_real=discriminator(imgs,conds)
            d_fake=discriminator(fake_imgs.detach(), conds)
            d_loss=criterion(d_real,real_labels)+ criterion(d_fake,fake_labels)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # TRAIN GENERATOR
            z =torch.randn(bs,z_dim).to(device)
            fake_imgs= generator(z,conds)
            d_fake=discriminator(fake_imgs,conds)
            g_loss=criterion(d_fake,real_labels)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
        print(f"Epoch [{epoch+1}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        torch.save({'epoch':epoch,'generator_state_dict':generator.state_dict(),'discriminator_state_dict':discriminator.state_dict(),
        'g_optimizer_state_dict':g_opt.state_dict(),'d_optimizer_state_dict':d_opt.state_dict()}, "checkpoint.pth")
        print(f"Saved checkpoint at epoch {epoch+1}") 
        if (epoch+1)%20==0:
            with torch.no_grad():
                sample=generator(torch.randn(1,z_dim).to(device),conds[0:1])
                #progress check (epoch 10/100 right now, looks like a gradient blob of color-this is taking forever)
                # epoch 20/100 kind of looks like a very very very very poor quality venusaur???
                # epoch 30/100 now looks like mickey mouse (took like 30 mins)
                #epoch 40 looks like goomba
                #epoch 50: grim reaper fish??
                #epoch 60: cerberus
                #epoch 70 looks like a huge fish
                #epoch 80: humanoid squid???
                #epoch 90 looks like zoroark
                #epoch 100: headless duck???- need to train more doesnt look realistic
                plt.imshow(sample[0].permute(1,2,0).cpu().numpy() *.5+.5)
                plt.title(f"Generated Image at Epoch {epoch+1}")
                plt.axis('off')
                plt.show()           

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#image = mpimg.imread(f'{merged_df.get("image_path").loc[2343]}')
#plt.imshow(image)
#plt.show()

dataset=PokemonDataset(merged_df)
dataloader=DataLoader(dataset,batch_size=32,shuffle=True)

sample_img,sample_condition=dataset[0]
#DIMENSION SHOUOLD BE 64
condition_dim=sample_condition.shape[0]

g= Generator(z_dim=100, condition_dim=condition_dim)
d=Discriminator(condition_dim=condition_dim)
start_epoch=0
checkpoint_path="checkpoint.pth"
g_opt=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
d_opt=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999))

if os.path.exists(checkpoint_path):
    checkpoint=torch.load(checkpoint_path)
    g.load_state_dict(checkpoint['generator_state_dict'])
    d.load_state_dict(checkpoint['discriminator_state_dict'])
    g_opt.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_opt.load_state_dict(checkpoint['d_optimizer_state_dict'])
    start_epoch=checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    print("No training checkpoint")
train_gan(g,d,dataloader,z_dim=100,condition_dim=condition_dim,start_epoch=start_epoch,g_opt=g_opt,d_opt=d_opt)