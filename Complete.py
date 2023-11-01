# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:29:33 2023

@author: hlato
"""

#First Static Visualizations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import pandas as pd
import matplotlib as mpl 
import scipy.io

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load the .csv file, skipping the first row (header) and specifying the delimiter
data = np.loadtxt(
    r'C:\Users\hlato\OneDrive\Escritorio\Thesis\Datos\7.9.23\Toma 1 Ventana\csv\20230907_16-20-54_dump_192.168.26_frame-1461.csv',
    delimiter=';',
    skiprows=1  # Skip the first row (header)
)


#--------------Here to show the point cloud plot for distances
# Extract X, Y, Z, and Intensity columns from the data (adjust column indices as needed)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
intensity = data[:, 3]  # Optional

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# You can adjust the point size and color based on intensity or other criteria
ax.scatter(x, y, z, s=1, c=intensity, cmap='viridis')

# Add labels and customize the plot as needed
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# Show the plot
plt.show()




#-------------Intensity Visualization
# Extract X, Y, Z, and Intensity columns from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
intensity = data[:, 4]  # Assuming intensity is in the 5th column (adjust if needed)

# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a colormap (you can choose any colormap you prefer)
cmap = plt.get_cmap('viridis')

# Normalize intensity values to the range [0, 1] for colormap mapping
intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

# Create the scatter plot with intensity-based coloring
scatter = ax.scatter(x, y, z, s=1, c=intensity_normalized, cmap=cmap)

# Add labels and customize the plot as needed
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Intensity Visualization')

# Add a color bar to show intensity values
cbar = plt.colorbar(scatter)
cbar.set_label('Intensity')

# Show the plot
plt.show()




#-----------------------------------
#-----------------------------------


#Visualization with Mayavi (Allows to navigate the point cloud in 3D) 
#Saves the file as.pcd


#-----------------------------------
#-----------------------------------


import numpy as np
from mayavi import mlab

# Load your point cloud data from the CSV file with the correct delimiter
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 2\Intento 2\20230907_16-20-54_dump_192.168.26_frame-1483 - Cloud-Test.csv'
data = np.genfromtxt(file_path, delimiter=';', skip_header=1)  # Correctly specify the delimiter and skip the header if present

# Extract X, Y, Z coordinates and normalized intensities from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
intensity = data[:, 3]  # Assuming intensity values are in the 5th column (adjust as needed)

# Normalize the intensity values to the range [0, 1]
#intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
intensity_normalized =intensity
# Create a Mayavi figure and visualize the point cloud with normalized intensities as color
mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
scatter = mlab.points3d(x, y, z, intensity_normalized, colormap='viridis', scale_mode='none', scale_factor=0.1)

# Adjust the view angle (you can rotate, pan, and zoom interactively)
mlab.view(azimuth=45, elevation=45, distance='auto')

# Add a color bar to show the intensity values
mlab.colorbar()

# Show the Mayavi plot
mlab.show()

#Save the file as .pcd
output_pcd_path = r'C:\Users\hlato\OneDrive\Escritorio\Thesis\Datos\7.9.23\Toma 1 Ventana\LAS\20230907_16-20-54_dump_192.168.26_frame-1461.pcd' 




#-----------------------
#------------------------

# POINT NET ARCHITECTURE Working

#------------------------
#------------------------




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import csv

# Load point cloud data from a CSV file
data = pd.read_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 2\frame1.csv', delimiter=';')

# Extract features (X) and labels (Y)
X = data.drop("Label", axis=1).values  # Features (X, Y, Z, INTENSITY)
Y = data["Label"].values  # Object labels for segmentation

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.int64)  # Use int64 for classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.int64)  # Use int64 for classification

# Create custom DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define PointNet architecture
class PointNet(nn.Module):
    num_classes = 7  # Number of material classes

    def __init__(self):
        super(PointNet, self).__init__()
        
        # Input layer (4 input features)
        self.input_layer = nn.Sequential(
            nn.Linear(4, 80),  # Adjust the number of neurons as needed
           #nn.Linear(4,64),  
            nn.ReLU()
        )
        # Hidden layers (Add more if necessary)
        self.hidden_layers = nn.Sequential(
            nn.Linear(80, 160),  # Adjust the number of neurons as needed
           #nn.Linear(64,128),
            nn.ReLU()
        )
        # Output layer (7 output neurons for multi-class classification)
        self.output_layer = nn.Sequential(
            nn.Linear(160, self.num_classes),
           #nn.Linear(128, self.num_classes), 
            nn.Softmax()  # Softmax activation for multi-class classification
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# Create the PointNet model
model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Use Cross-Entropy Loss for classification

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_Y)
        loss.backward()
        optimizer.step()
        
              
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

# Evaluate the trained model on test data (optional)
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)

# Visualize segmented objects using Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X_test[:, :3])  # Use original XYZ coordinates

# Create a color map for visualization
num_classes = 7
color_map = [
    [1, 0, 0],  # Class 0: Red
    [0, 1, 0],  # Class 1: Green Mansory
    [0, 0, 1],  # Class 2: Blue Concrete
    [1, 1, 0],  # Class 3: Yellow Steel
    [0, 1, 1],  # Class 4: Cyan Metal
    [1, 0, 1],  # Class 5: Magenta Noise
    [0.5, 0.5, 0.5],  # Class 6: Gray (for other or unknown)
]

# Ensure that predicted labels are within the valid range
valid_predicted_labels = [label % num_classes for label in predicted_labels]

# Create an array of colors based on predicted labels
point_colors = np.array([color_map[label] for label in valid_predicted_labels])

# Assign colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(point_colors)

#-------new section
# Create a custom legend as a point cloud with colored spheres
legend_pcd = o3d.geometry.PointCloud()
legend_colors = np.array([color_map[i] for i in range(num_classes)])  # Use colors for known classes
legend_pcd.points = o3d.utility.Vector3dVector(np.zeros((num_classes, 3)))  # Create positions for spheres
legend_pcd.colors = o3d.utility.Vector3dVector(legend_colors)
legend_pcd.translate(np.array([1, 0, 0]))  # Adjust the position of the legend

# Combine the legend with the point cloud
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(pcd)
visualizer.add_geometry(legend_pcd)
visualizer.run()
visualizer.destroy_window()





#-----------



# Visualize the segmented objects
o3d.visualization.draw_geometries([pcd]) 





#---------Save the Epcoh and Loss





#---------------
# HISTOGRAMS
#---------------


import pandas as pd
import matplotlib.pyplot as plt


# Replace 'your_file.csv' with the path to your CSV file
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\frame-1685-results.csv'
# Specify the custom delimiter
delimiter = ';'
# Load the CSV file using the custom delimiter
df = pd.read_csv(file_path, delimiter=delimiter)


print(df.columns)


# Create a histogram for intensity values
plt.figure(figsize=(10, 6))
plt.hist(df['INTENSITY'].astype(float), bins=20, color='blue', alpha=0.7)
plt.title('Intensity Histogram for Location 4')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#------------------
# Previuos Histogram With colors an Labels
#------------------


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\frame-1685-results.csv'

# Specify the custom delimiter
delimiter = ';'

# Load the CSV file using the custom delimiter
df = pd.read_csv(file_path, delimiter=delimiter)



# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Plastic
    # 04 : Noise
    # 05 : Glass
    # 06 : Mansory
    # 07 : Wood



# Define label colors manually
label_colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'yellow',
    4: 'cyan',
    5: 'magenta',
    6: 'gray',
    7: 'purple'
}

# Define material names corresponding to label colors
material_names = {
    'red': 'Empty',
    'green': 'Concrete',
    'blue': 'Metal',
    'yellow': 'Stone',
    'cyan': 'Noise',
    'magenta': 'Glass',
    'gray': 'Mansory' ,
    'purple': 'Wood'
}




# Create a histogram for each label with custom colors and material names
plt.figure(figsize=(10, 6))
for label, color in label_colors.items():
    plt.hist(
        df[df['Label'].astype(int) == label]['Label'].astype(int),
        bins=range(0, df['Label'].astype(int).max() + 2),
        align='left',
        color=color,
        alpha=0.7,
        label=material_names[color]  # Use material names as labels
    )

# Create a legend with material names as labels
legend_labels = [material_names[color] for color in label_colors.values()]
plt.legend(legend_labels, loc='best')

plt.title('Material Histogram for Location 4 Results')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#------------------Ground truth label histogram

import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\frame-1685-results.csv'

# Specify the custom delimiter
delimiter = ';'

# Load the CSV file using the custom delimiter
df = pd.read_csv(file_path, delimiter=delimiter)

# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Plastic
    # 04 : Noise
    # 05 : Glass
    # 06 : Mansory
    # 07 : Wood



# Define label colors manually
label_colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'yellow',
    4: 'cyan',
    5: 'magenta',
    6: 'gray',
    7: 'purple'
}

# Define material names corresponding to label colors
material_names = {
    'red': 'Empty',
    'green': 'Concrete',
    'blue': 'Metal',
    'yellow': 'Stone',
    'cyan': 'Noise',
    'magenta': 'Glass',
    'gray': 'Mansory' ,
    'purple': 'Wood'
}

# Create a histogram for each label with custom colors and material names
plt.figure(figsize=(10, 6))
for label, color in label_colors.items():
    plt.hist(
        df[df['Ground Truth Label'].astype(int) == label]['Ground Truth Label'].astype(int),
        bins=range(0, df['Ground Truth Label'].astype(int).max() + 2),
        align='left',
        color=color,
        alpha=0.7,
        label=material_names[color]  # Use material names as labels
    )

# Create a legend with material names as labels
legend_labels = [material_names[color] for color in label_colors.values()]
plt.legend(legend_labels, loc='best')

plt.title('Material Histogram for Location 4')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()













