#! /usr/bin/env python3

#Proyecto Sistemas de Comunicaciones Primavera - 2024
#Integrantes: Gabriel Ona, Emilia Casares y Jose Montaguano

### Prediction 0 -> Retro
### Prediction 1 -> Paralelo
### Prediction 2 -> Diagonal
### Prediction 3 -> Frente

#ROS LIBRARIES
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_msgs.msg import Bool

#MACHINE LEARNING MODEL LIBRARIES- NEURAL NETWORK
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#DATA MANAGEMENT LIBRARIES
import pandas as pd
import numpy as np
import os
import subprocess
subprocess.run("export TORCH_NNPACK=0", shell=True)

#Paths
scripts_dir = os.path.dirname(os.path.realpath(__file__))
scripts_dir = scripts_dir.replace('/scripts','',-1)
path_feature_selection = os.path.join(scripts_dir, 'model/k_best_selected_features', 'selected_features.csv')
path_nn_model = os.path.join(scripts_dir, 'model/pretrained_model', 'model_final_cpu.pth')

#Variables
featured_index = pd.read_csv(path_feature_selection)
featured_index = featured_index.iloc[:].values.astype(dtype=np.int32)
input_layer_length = len(featured_index)
output_classes = 4
parking_enable = False
input_data = None

#Publication
pub = rospy.Publisher('/parking/type', String, queue_size=1)

#Neural Network Architecture
class AlexNet1D(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
#Neural Network Loader
def nn_model_load(output_classes):
    model = AlexNet1D(output_classes)
    model.load_state_dict(torch.load(path_nn_model))
    return model

#Evaluation 
def evaluation(model,device,x):
    model.eval()
    with torch.no_grad():
        x_eval = x.clone().detach().requires_grad_(True)
        return model(x_eval)
    
#Message Display
def take_action(input_data):
    global parking_enable
    if (parking_enable):
        prediction = evaluation(model,device,input_data)
        print("")
        print(" ---------------------------- PARQUEANDO  ---------------------------- ")
        print(" DETECTANDO ENTORNO...")
        msg = String()
        if (torch.argmax(prediction).tolist() == 0):
            msg.data = "retro"
        elif(torch.argmax(prediction).tolist() == 1):
            msg.data = "paralelo"
        elif(torch.argmax(prediction).tolist() == 2):
            msg.data = "diagonal"
        elif(torch.argmax(prediction).tolist() == 3):
            msg.data = "frente"

        print("")
        print("DETECCION COMPLETADA :)")
        print("")
        print("El automovil se encuentra en un parqueo tipo: " + str(msg.data))
        print("Realizando algoritmo de parqueo " + str(msg.data))
        print(" --------------------------------------------------------------------- ")

        pub.publish(msg)
        parking_enable = False

#Callback Lidar Sensor
def callback_laser(msg):
    global input_data
    lidar_range_min = msg.range_min
    lidar_range_max = msg.range_max
    input_data = []
    for index in featured_index:
        value = 0
        if (msg.ranges[int(index)] == float('inf')):
            value = 0
        else:
            value = msg.ranges[int(index)]

        x = (value - lidar_range_min)/(lidar_range_max - lidar_range_min)       #Min Max Normalization
        input_data.append(x)
    input_data = torch.tensor(input_data).view(-1, input_layer_length).to(device)
    input_data = input_data.unsqueeze(0)
    take_action(input_data)

#Callback Parking Enable
def callback_parking_enable(msg):
    global parking_enable
    if (msg.data == True):
        parking_enable = True
 
#Pytorch Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn_model_load(output_classes)
model.to(device)

def main():
    global parking_enable
    global input_data
    
    sub_parking_enable = rospy.Subscriber('/parking/enable', Bool, callback_parking_enable)
    sub_input_data = rospy.Subscriber('/scan', LaserScan, callback_laser)
    
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('ml_node')
    rospy.sleep(5)
    main()