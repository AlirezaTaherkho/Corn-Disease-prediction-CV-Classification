# data aug model was the best model

import cv2 as cv
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import models
from torch import nn



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2))

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*10*10,256),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.fc3 = nn.Sequential(
            nn.Linear(64,4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def model_format(img):
    mean = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
    std = torch.tensor([0.5,0.5,0.5]).view(3,1,1)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (200,200))
    img = torch.from_numpy(img)

    img = img.float()
    img = img / 255.0

    img = img.permute(2,0,1)
    img = (img - mean) / std
    img = img.unsqueeze(0)

    return img



if __name__ == "__main__":
    m = Net()
    m.load_state_dict(torch.load(r"C:\Users\wwwat\Desktop\Corn Leaf Disease\data aug model\data_aug_model.pth", map_location="cpu"))
    m.eval()

    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    cap = cv.VideoCapture(0)

    while True:
        rec, frame = cap.read()
        if not rec:
            break

        frame = cv.flip(frame, 1)

        input_img = model_format(frame)

        with torch.inference_mode():
            output = m(input_img)
            prediction = torch.argmax(output, dim=1)

            cv.putText(frame, classes[prediction.item()], (40,40), cv.FONT_HERSHEY_PLAIN, 3, (150,0,0), 3)
            
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv.destroyAllWindows()





























