import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import h5py
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import AgeGenderModel
import time
from data_loader import data_loader, data_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

def main(args):
    hf = h5py.File(args.h5py_path)
    images = hf['images']
    age = hf['age']
    gender = hf['Gender']

    age_dict = {0: '0-25', 1:'25-50', 2:'50-75', 3:'75-'}
    gender_dict = {0 : 'Male', 1: 'Female'}
    image_ids = list(range(0, len(images)))

    random.Random(4).shuffle(image_ids)
    train_ids = image_ids[:int(np.floor(args.split_size*len(image_ids)))]
    train_ids.sort()
    valid_ids = image_ids[int(np.floor(args.split_size*len(image_ids))):]
    valid_ids.sort()

    train_images, train_age, train_gender  = data_split(train_ids, hf)
    valid_images, valid_age, valid_gender  = data_split(valid_ids, hf)

    train_transformations = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.70832765, 0.5542221 , 0.487289), (0.22306082, 0.22076079, 0.22623643))
    ])
    valid_transformations = transforms.Compose([
        transforms.CenterCrop(args.crop_size),                                     
        transforms.ToTensor(),
        transforms.Normalize((0.70832765, 0.5542221 , 0.487289), (0.22306082, 0.22076079, 0.22623643))
    ])

    train_loader = data_loader(train_images,train_age, train_gender, transform= train_transformations, batch_size = args.batch_size, shuffle= True)
    valid_loader = data_loader(valid_images,valid_age, valid_gender, transform= valid_transformations, batch_size = args.batch_size, shuffle= False)


    model = AgeGenderModel(2, 4)
    model = model.to(device)
    gender_loss = nn.CrossEntropyLoss().to(device)
    age_loss = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs
    f = 0
    train_loss_age = []
    train_loss_gender = []
    valid_loss_age = []
    valid_loss_gender = []
    for epoch in range(num_epochs):
    if f < 5:
        train_loss_g = 0
        train_loss_v = 0  
        valid_loss_g = 0
        valid_loss_v = 0
        valid_loss_a = 0
        valid_loss_min = 10000
        model.train()
    #  model = model.double()
        start = time.time()
        for img, age, gender in train_loader:
        #img, age, gender = img.type(torch.DoubleTensor),
        gender = gender.type(torch.LongTensor)
        age = age.type(torch.LongTensor) 
        optimizer.zero_grad()
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
    
        gender_p, age_p = model(img)
        gender_l = gender_loss(gender_p, gender)
        #print(gender_l)
        age_l = age_loss(age_p, age)
        #print(age_l)
        loss = 0.3*gender_l+0.7*age_l
        loss.backward()
        optimizer.step()
        train_loss_g += gender_l.item()
        train_loss_v += age_l.item()
        train_loss_age.append(train_loss_v/len(train_loader))
        train_loss_gender.append(train_loss_g/len(train_loader))
        print('Epoch:  ' + str(epoch) + '        train loss_gender:  ' + str(train_loss_g/len(train_loader))+ '        train_loss_age:  ' + str(train_loss_v/len(train_loader)))

        model.eval()
        for img_v, age_v, gender_v in valid_loader:
        gender_v = gender_v.type(torch.LongTensor)
        age_v = age_v.type(torch.LongTensor) 
        img_v = img_v.to(device)
        age_v = age_v.to(device)
        gender_v = gender_v.to(device)
        gender_pv, age_pv = model(img_v)
        gender_lv = gender_loss(gender_pv, gender_v)
        age_lv = age_loss(age_pv, age_v)
        valid_loss_g += gender_lv.item()
        valid_loss_a += age_lv.item()
        valid_loss_v = valid_loss_g+valid_loss_a
        if valid_loss_v < valid_loss_min:
        valid_loss_min = valid_loss_v
        f = 0
        print('saving model')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        else:
        f+=1
        valid_loss_age.append(valid_loss_a/len(valid_loader))
        valid_loss_gender.append(valid_loss_g/len(valid_loader))
        print('Epoch:  ' + str(epoch) + '        valid loss_gender:  ' + str(valid_loss_g/len(valid_loader))+ '        valid_loss_age:  ' + str(valid_loss_a/len(valid_loader)))




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--h5py_path', type=str, default='data/resized2014', help='directory for h5py images')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=4)
    parser.add_argument('--split_size', type=float, default=0.75)
    args = parser.parse_args()
    print(args)
    main(args)