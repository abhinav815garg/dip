# GLCM-Based Texture Classification
# --------------------------------
# This script:
# 1. Loads a grayscale texture dataset
# 2. Preprocesses images
# 3. Computes GLCM matrices
# 4. Extracts texture features
# 5. Trains a classifier
# 6. Evaluates performance

# Install required libraries if needed:
# pip install numpy pandas matplotlib scikit-image scikit-learn opencv-python

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# DATASET PATH
# =========================================================
# Dataset structure should be:
#
# dataset/
# ├── class1/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── class2/
# ├── class3/
#
# Change this path to your dataset folder

dataset_path = "dataset"

# =========================================================
# PARAMETERS
# =========================================================

IMAGE_SIZE = (128, 128)

# GLCM parameters
distances = [1, 2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# =========================================================
# FEATURE EXTRACTION FUNCTION
# =========================================================

def extract_glcm_features(image):

    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []

    # Extract texture features
    properties = [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation',
        'ASM'
    ]

    for prop in properties:
        values = graycoprops(glcm, prop)

        # Mean and std of each property
        features.extend(values.flatten())

    return features

# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []

classes = os.listdir(dataset_path)

print("Loading dataset...")

for label in classes:

    class_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        img_path = os.path.join(class_folder, file)

        # Read image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, IMAGE_SIZE)

        # Extract GLCM features
        features = extract_glcm_features(image)

        X.append(features)
        y.append(label)

print("Dataset loaded successfully.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE NORMALIZATION
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# TRAIN CLASSIFIER
# =========================================================

# -------- Option 1: SVM --------
classifier = SVC(kernel='rbf')

# -------- Option 2: KNN --------
# classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = classifier.predict(X_test)

# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# VISUALIZE CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Write values inside matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha='center',
                 va='center',
                 color='red')

plt.tight_layout()
plt.show()

# =========================================================
# SAVE FEATURES TO CSV (OPTIONAL)
# =========================================================

feature_df = pd.DataFrame(X)
feature_df['label'] = y

feature_df.to_csv("glcm_features.csv", index=False)

print("\nFeature dataset saved as glcm_features.csv")