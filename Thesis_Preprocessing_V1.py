# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:29:34 2023

@author: hlato
Technical University of Munich
TUM School of Engineering and Design
Lehrstuhl für Computergestützte Modellierung und Simulation

"""

#                               FINAL CODE FOR POINT CLOUD SEGMENTATION USING INTENSITY VALUES

# CHAPTER 1: Data Preprocessing
# Steps:
# 1.1. Upload the raw dataset in ASCII .csv format to CloudCompare.
# 1.2. Clean the Point Cloud in CloudCompare and export it into .csv file and .pcd file
# 1.3. In Matlab Lidar labeler check the Point Cloud and assign manually the labels to the surfaces
# 1.4. Export the VoxelLabelData in .mat format
# 1.5. Use the code bellow to convert from .mat to .csv


# A). ------ CONVERT FROM .mat TO .csv--#
#-----------------------------------#

import scipy.io as sio
import pandas as pd
# Load the .mat file
data = sio.loadmat(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\WithGlass\VoxelLabelData\Main_entrance.mat')
# Get the point cloud scan
L = data['L']
# Take the values of all columns
X = L[:, 0]
Y = L[:, 1]
Z = L[:, 2]
Label = L[:, 3]
# Create a DataFrame with the column structure
df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'Label': Label})
# Export the file in .csv format
df.to_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\WithGlass\VoxelLabelData\Main_entrance.csv', index=False, sep=',')





# B). --------VISUALIZE CONVERTED POINT CLOUD FOR QUALITY CHECK-----#
#-------------------------------------------------------------------#

# Only XYZ point coordinates
import numpy as np
from mayavi import mlab

# Load point cloud data from the CSV file with the correct delimiter - In some cases ; or , 
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 6 Metal Building TUM\Intento 2\VoxelLabelData\frame-1632-matlabel.csv'
data = np.genfromtxt(file_path, delimiter=';', skip_header=1)  # Correctly specify the delimiter and skip the header if present

# Extract X, Y, Z coordinates from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a Mayavi figure and visualize the point cloud
mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
scatter = mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')
mlab.view(azimuth=45, elevation=45, distance='auto')
mlab.show()
#-------------------------


# Visualization with Surface Intensities
import numpy as np
from mayavi import mlab
import open3d as o3d 

# Load your point cloud data from the CSV file with the correct delimiter
file_path = r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 5 Main Building From Flags location\Intento 2\20231010_09-58-44_dump_192.168.26_frame-662 - Cloud-results.csv'
data = np.genfromtxt(file_path, delimiter=';', skip_header=1)  # Correctly specify the delimiter and skip the header if present

# Extract X, Y, Z coordinates and normalized intensities from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
intensity = data[:, 3]  # Assuming intensity values are in the 4th column (adjust as needed)
labeled = data[:,4]

# Normalize the intensity values to the range [0, 1]
#intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

# Create a Mayavi figure and visualize the point cloud with normalized intensities as color
mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
#scatter = mlab.points3d(x, y, z, intensity_normalized, colormap='viridis', scale_mode='none', scale_factor=0.1)  # Here uses the normalized intensities
#scatter = mlab.points3d(x, y, z, intensity, colormap='viridis', scale_mode='none', scale_factor=0.1) # Here uses the raw intensities
scatter = mlab.points3d(x, y, z, labeled, colormap='viridis', scale_mode='none', scale_factor=0.1) # Here visualizes the labels of the materials


# Adjust the view angle for playing around with the visualization window
mlab.view(azimuth=45, elevation=45, distance='auto')

# Add a color bar to show the intensity values
mlab.colorbar()

# Show the Mayavi plot
mlab.show()
#---------------------------------

# 1.6. Combine manually the Exported .csv from CloudCompare and the previuos labeled .csv from Matlab Lidar Labeler
# 1.7. Using a new dataset for testing, repeat steps 1.1. and 1.2.
#             1.1. Upload the -test- raw dataset in ASCII .csv format to CloudCompare.
#             1.2. Clean the -test- Point Cloud in CloudCompare and export it into .csv file 

#---------------------------------


# CHAPTER 2: POINTNET ARCHITECTURE FOR SEGMENTATION
#                  USING POINT CLOUD INTENSITIES


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import matplotlib.pyplot as plt

# Load point cloud data from a CSV file for train the model from the section  1.5. A)
data = pd.read_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\20231010_09-40-27_dump_192.168.26_frame-1685 - Cloud.csv', delimiter=';')

# Extract features (X) and labels (Y)
X_train = data.drop("Label", axis=1).values  # Features (X, Y, Z, INTENSITY)
Y_train = data["Label"].values  # Object labels for segmentation

# Standardize features from the previuos point cloud
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.int64)  # Use int64 for classification


# Create custom DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



#Count the number of unique classes in the dataset
num_classes = len(data["Label"].unique())

#check the num_classes identified in the labeled file.

# Define PointNet architecture
class PointNet(nn.Module):
    #num_classes = 5  # Number of material classes

    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        # Input layer (4 input features)
        self.input_layer = nn.Sequential(
            nn.Linear(4, 64),  # Adjust the number of neurons as needed
            nn.ReLU()
        )
        # Hidden layers (Add more if necessary)
        self.hidden_layers = nn.Sequential(
            nn.Linear(64, 128),  # Adjust the number of neurons as needed
            nn.ReLU()
        )
        # Output layer (7 output neurons for multi-class classification)
        self.output_layer = nn.Sequential(
            nn.Linear(128, self.num_classes),
            nn.Softmax()  # Softmax activation for multi-class classification
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


# Create the PointNet model - CROSS ENTROPY LOSS CRITERION ------
# Be aware about the num_classes.
model = PointNet(num_classes=7)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Use Cross-Entropy Loss for classification



# Training loop
num_epochs = 100
losses = [] #Initialize an empty list to store the loss values
epochs =[] #Initialize an empy list to store the epochs

#Remember: For storing the arrays epochs [] and loss[] use the segment of the SaveCopy.



# Plot the loss over epochs - better to use the SaveCopy plotting section
plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid(True)
plt.show()


# Save the trained model 
torch.save(model.state_dict(),'pointnet_model1.pth')

# Load the previuos saved model 
loaded_model = PointNet(num_classes=7)  # Create a model with the same architecture 
loaded_model.load_state_dict(torch.load('pointnet_model1.pth'))
loaded_model.eval()  # Set the model to evaluation mode

#Load the testing data from a different CSV file
testing_data =pd.read_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\20231010_09-40-27_dump_192.168.26_frame-1685 - Cloud -SC.csv', delimiter=';')

#Extract features (X, Y, Z, INTENSITY) for intensity
X_test = testing_data[['X','Y','Z','INTENSITY']].values

#Standarize features
X_test = scaler.transform(X_test)

#Convert the testing data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Evaluate the trained model on test data (optional)
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)

# Add the predicted labels as a new 'Label' in the testing data
testing_data['Label'] = predicted_labels.numpy()

#Save the testing data with predicted labels
testing_data.to_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\frame-1685-results.csv', index=False)


#---------------------


# CHAPTER 3: mIOU 
# In the testing_data file, paste the Ground Truth Label from the training Labeled file. Save it and Load it again.
# Check the number of rows between the Pasted Ground Truth and the Labeled File. Should be the same.


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, jaccard_score


testing_data = pd.read_csv(r'C:\Users\hlato\OneDrive\Escritorio\Preprocessed\Train\Location 4 Main Entrance Arcistrasse\frame-1685-results.csv', delimiter=';')


# Upload the ground truth as "y_true" and predicted labels from results as "y_pred"
y_true = testing_data['Ground Truth Label']  # Replace 'Ground_Truth_Label' with the actual column name
y_pred = testing_data['Label']  # 'Label' contains the predicted labels

# Calculate the confusion matrix
confusion = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(confusion)



# Calculate mIOU (mean Intersection over Union)
iou = []
for i in range(len(confusion)):
    true_positive = confusion[i, i]
    false_positive = sum(confusion[:, i]) - true_positive
    false_negative = sum(confusion[i, :]) - true_positive
    iou.append(true_positive / (true_positive + false_positive + false_negative))

mIOU = sum(iou) / len(iou)

print(f"Mean Intersection over Union (mIOU): {mIOU}")

#----------------------------
# Option 2 for mIOU and reporting
#

# Ground truth labels
ground_truth = testing_data['Ground Truth Label'].values

#Predicted labels
predicted_labels=testing_data['Label'].values # This is used only when checking separate files

# Calculate mIOU
mIOU = jaccard_score(ground_truth, predicted_labels.numpy(), average='macro')
mIOU = jaccard_score(ground_truth, predicted_labels, average='macro') # This is used only when checking separate files
print(f"Mean IOU: {mIOU}")

print("Classification Report:")
print(classification_report(ground_truth, predicted_labels.numpy()))
print(classification_report(ground_truth, predicted_labels))# This is used only when checking separate files


# Create a confusion matrix
cm = confusion_matrix(ground_truth, predicted_labels.numpy())
cm = confusion_matrix(ground_truth, predicted_labels)# This is used only when checking separate files

# Print and visualize the confusion matrix
print("Confusion Matrix:")
print(cm)


iou = []
for i in range(len(cm)):
    true_positive = cm[i, i]
    false_positive = sum(cm[:, i]) - true_positive
    false_negative = sum(cm[i, :]) - true_positive
    iou.append(true_positive / (true_positive + false_positive + false_negative))

mIOU = sum(iou) / len(iou)

print(f"Mean Intersection over Union (mIOU): {mIOU}")



print("Classification Report:")
print(classification_report(ground_truth, predicted_labels.numpy()))
print(classification_report(ground_truth, predicted_labels))# This is used only when checking separate files

#-------------------------




# CHAPTER 4: RESULTS
#------------------------
#------------------------ RESULTS FOR THE DIFFERENT SURFACES AND MATERIALS
#------------------------

# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Plastic
    # 04 : Noise
    # 05 : Glass
    # 06 : Mansory
    # 07 : Wood


# 4.1. Location 6: Metal Building TUM 

# Number of Epochs:  1000
#             Loss:~ 0.0015602267 


# Mean IOU: 0.1829914130708152


# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.00      0.00      0.00         5
#            2       0.91      1.00      0.96     25466
#            4       0.00      0.00      0.00      1470
#            5       0.00      0.00      0.00         2
#            7       0.00      0.00      0.00       890
#
#     accuracy                           0.91     27833
#    macro avg       0.18      0.20      0.19     27833
# weighted avg       0.84      0.91      0.87     27833



# Confusion Matrix:
# [[    0     5     0     0     0]
#  [    0 25466     0     0     0]
#  [    0  1470     0     0     0]
#  [    0     2     0     0     0]
#  [    0   890     0     0     0]]


#---------------------------------------------------------
#------------*End of results Location 6*------------------


# 4.2 Location 1: University Main Campus - Interior Scan

# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Plastic
    # 04 : Noise
    # 05 : Glass
    # 06 : Mansory
    # 07 : Wood

#Mean Intersection over Union (mIOU): 0.29021105049825363


#Confusion Matrix:
#[[    2   293    90   123     0     0]
# [   23 30428  1046  1527     0     0]
# [    1   586   692     1     0     0]
# [   36  1430    41  4931     0     0]
# [    0    12    19     0     0     0]
# [    0    22   108     0     0     0]]


# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.03      0.00      0.01       508
#            1       0.93      0.92      0.92     33024
#            2       0.35      0.54      0.42      1280
#            3       0.75      0.77      0.76      6438
#            4       0.00      0.00      0.00        31
#            5       0.00      0.00      0.00       130
#
#     accuracy                           0.87     41411
#    macro avg       0.34      0.37      0.35     41411
# weighted avg       0.87      0.87      0.87     41411


#---------------------------------------------------------
#------------*End of results Location 1*------------------


#4.3 Location 2: Residential buildings Augsburg


# Number of Epochs:  1000
#             Loss:~ 0.00158282720107781



#Mean Intersection over Union (mIOU): 0.31513632957920157

# Confusion Matrix:
# [[    0   267   128   101     0     0    23]
#  [    0 18126   428   108     0     0    78]
#  [    0   565   686   297     0     0    75]
#  [    0   123   359   502     0     0     0]
#  [    0   146     0     0     0     0     0]
#  [    0   135     0     0     0     0     0]
#  [    0   614    71     1     0     0  2284]]




#Classification Report:
    
#               precision    recall  f1-score   support
#
#            0       0.00      0.00      0.00       519
#            1       0.91      0.97      0.94     18740
#            2       0.41      0.42      0.42      1623
#            3       0.50      0.51      0.50       984
#            4       0.00      0.00      0.00       146
#            5       0.00      0.00      0.00       135
#            6       0.93      0.77      0.84      2970
#
#     accuracy                           0.86     25117
#    macro avg       0.39      0.38      0.39     25117
# weighted avg       0.83      0.86      0.84     25117

#---------------------------------------------------------
#------------*End of results Location 2*------------------

#4.4. Location 3 Pinakothek



# Materials:
    # 00 : None
    # 01 : Stone
    # 05 : Glass
    # 06 : Mansory



# Number of Epochs: 1000
#             Loss:~0.0011316236922557098


# Mean Intersection over Union (mIOU): 0.3938345743233276


#Confusion Matrix:
#[[    0    14     0    24]
# [    0  5393     0  1288]
# [    0     0     0   160]
# [    0  1302     0 25235]]


#Classification Report:
    
#               precision    recall  f1-score   support
#
#            0       0.00      0.00      0.00        38
#            1       0.80      0.81      0.81      6681
#            5       0.00      0.00      0.00       160
#            6       0.94      0.95      0.95     26537
#
#     accuracy                           0.92     33416
#    macro avg       0.44      0.44      0.44     33416
# weighted avg       0.91      0.92      0.91     33416

#---------------------------------------------------------
#------------*End of results Location 3*------------------


#4.5 Location 4 TUM Main Entrance Arcistraße


# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Stone
    # 04 : Noise
   


# Epochs: 1000 
#   Loss:~0.0012692614640065756


#Confusion Matrix:
# [[  340   142    17  1269     2]
#  [  147   401   161  2232    35]
#  [   10   160   102   698    18]
#  [ 1027  1639   610 14349   131]
#  [   20    45    15   388     4]]



#Mean Intersection over Union (mIOU): 0.1800766690766274


#Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.22      0.19      0.21      1770
#            1       0.17      0.13      0.15      2976
#            2       0.11      0.10      0.11       988
#            3       0.76      0.81      0.78     17756
#            4       0.02      0.01      0.01       472
#
#     accuracy                           0.63     23962
#    macro avg       0.26      0.25      0.25     23962
# weighted avg       0.60      0.63      0.62     23962

#---------------------------------------------------------


# Results corrected



#Epoch [100/100] Loss: 0.0018068648646687316
#Confusion Matrix:
#[[ 1433    61     9   267     0     0]
# [    7  1707     0  1262     0     0]
# [    0     0   988     0     0     0]
# [    0    19     0 15347     0     0]
# [    0   204     0   268     0     0]
# [    0     0     0  2390     0     0]]

#Mean Intersection over Union (mIOU): 0.5176500335401063


#Classification Report:
    
    
#               precision    recall  f1-score   support
#
#            0       1.00      0.81      0.89      1770
#            1       0.86      0.57      0.69      2976
#            2       0.99      1.00      1.00       988
#            3       0.79      1.00      0.88     15366
#            4       0.00      0.00      0.00       472
#            5       0.00      0.00      0.00      2390
#
#     accuracy                           0.81     23962
#    macro avg       0.60      0.56      0.58     23962
# weighted avg       0.72      0.81      0.76     23962









#------------*End of results Location 4*------------------

#4.6 Location 5 Main Building


# Materials:
    # 01 : Concrete
    # 02 : Metal
    # 03 : Plastic
    # 04 : Noise
    # 05 : Glass
    # 06 : Mansory
    # 07 : Wood


# Epochs: 1000
#   Loss:~ 0.0017162210922696242


#Confusion Matrix:
#[[    0    75     0     0     0     0     0]
# [    0 23116     0     0     0     0     0]
# [    0   415     0     0     0     0     0]
# [    0   260     0     0     0     0     0]
# [    0   102     0     0     0     0     0]
# [    0   454     0     0     0     0     0]
# [    0  1039     0     0     0     0     0]]


#Mean Intersection over Union (mIOU): 0.12969976490655175

#Classification Report:


#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00        75
#            1       0.91      1.00      0.95     23116
#            2       0.00      0.00      0.00       415
#            3       0.00      0.00      0.00       260
#            4       0.00      0.00      0.00       102
#            5       0.00      0.00      0.00       454
#            7       0.00      0.00      0.00      1039

#     accuracy                           0.91     25461
#    macro avg       0.13      0.14      0.14     25461
# weighted avg       0.82      0.91      0.86     25461


#---------------------------------------------------------
#------------*End of results Location 5*------------------















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

#-------Section for visualization
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





















