import cv2 as cv
import matplotlib.pyplot as plt
from model import AgeGenderModel
from torchvision import transforms
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

age_dict = {0: '0-25', 1:'25-50', 2:'50-75', 3:'75-'}
gender_dict = {0 : 'Male', 1: 'Female'}

def main(args):
    face_cascade = cv.CascadeClassifier('/content/drive/My Drive/haarcascades/haarcascade_frontalface_default.xml')
    img = cv.imread(args.image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x,y,w,h = faces[0]
    im = img[x-20:x+w+20, y-20:y+h+20]
    plt.imshow(img)
    model = AgeGenderModel(2, 4)
    model = model.to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    im = Image.fromarray(im)
    im = im.resize(args.resize,args.resize)
    im_ = transforms.ToTensor()(im)
    im_ = transforms.Normalize((0.70832765, 0.5542221 , 0.487289), (0.22306082, 0.22076079, 0.22623643))(im_)
    im_ = im_.unsqueeze(0)
    im_ = im_.to(device)
    gender,age = model(im_)

    gender, age = gender_img(img_path)

    print('gender:    '+  gender_dict[gender.argmax(dim=1).tolist()[0]])
    print('age:       '+ age_dict[age.argmax(dim=1).tolist()[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='images/running.jpg', help='input image for predicting Age and Gender')
    parser.add_argument('--model_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained model')
    parser.add_argument('--resize', type=int, default=224, help = 'image resize')