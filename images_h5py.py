from PIL import Image
import h5py
import os
import numpy as np

image_path = 'UTKFace/'
files = os.listdir(image_path)

images = []
age =[] 
gender = []
i = 0
for f in files:
    img = Image.open('UTKFace/'+f)
    images.append(np.array(img))
    age.append(f.split('_')[0])
    gender.append(f.split('_')[1])

images = np.array(images)
age = np.array(age).astype(int)
#age = np.expand_dims(age, axis=1)
gender = np.array(gender).astype(int)

hf = h5py.File('data.h5', 'w')
hf.create_dataset('images', data=images)
hf.create_dataset('age', data=age)
hf.create_dataset('Gender', data = gender)
hf.close()