from torchvision import models
import torch.nn as nn

class AgeGenderModel(nn.Module):
  def __init__(self, output_size, age_output, fine_tune=False, dropout = 0.4):
    super(AgeGenderModel, self).__init__()
    self.output_size = output_size
    self.age_outupt = age_output
    #elf.age_dict = age_dict
    self.resnet152 = models.resnet101(pretrained=True)
    num_ftrs = self.resnet152.fc.in_features
    for p in self.resnet152.parameters():
        p.requires_grad = fine_tune
    self.resnet152.fc = nn.Identity()
    self.fc1 = nn.Linear(num_ftrs, output_size)
    self.fc2 = nn.Linear(num_ftrs, age_output)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    x = self.resnet152(x)
    gender = self.fc1(x)
    #g = g.unsqueeze(-1).float()
    #gender_ = gender.argmax(dim=1)
    #gender_ = gender_.unsqueeze(-1).float()
    x = self.dropout(x)
    age = self.fc2(x)
    return gender, age