import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

def data_split(ids, hf):
    return hf['images'][ids], hf['age'][ids], hf['Gender'][ids]

class ImageDataset(Dataset):
  def __init__(self, images, age, gender, transform=None):
    self.images = images
    self.transform = transform
    self.age = age
    self.gender = gender
  
  def __getitem__(self, idx):
    img =  self.images[idx]
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((224, 224))
    #img = np.array(img)
    age = self.age[idx]
    if age >=0 and age <25:
      age = 0
    elif age >=25 and age < 50:
      age = 1
    elif age >=50 and age < 75:
      age = 2
    else:
      age = 3
    Gender = self.gender[idx]
    if self.transform:
      img = self.transform(img)
    
    return img, age, Gender
  
  def __len__(self):
    return len(self.age)

def data_loader(images, age, gender, transform, batch_size, shuffle):
    imageloader = ImageDataset(images,age, gender, transform= transform)
    return DataLoader(imageloader,batch_size=batch_size,shuffle=shuffle)

