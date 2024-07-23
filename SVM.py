import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cats_directory = r"C:\Users\PRAGYAN\Desktop\ProdigyProjects\train"
dogs_directory = r"C:\Users\PRAGYAN\Desktop\ProdigyProjects\test1"

# Function to load images and assign labels
def fetch_images_and_labels(directory, label):
    images, labels = [], []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            img_path = os.path.join(directory, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (64, 64))
            images.append(resized_image)
            labels.append(label)
    return images, labels

# Load cat images and assign label 0
cat_images, cat_labels = fetch_images_and_labels(cats_directory, 0)

# Load dog images and assign label 1
dog_images, dog_labels = fetch_images_and_labels(dogs_directory, 1)

# Combine cat and dog data
all_images = np.array(cat_images + dog_images)
all_labels = np.array(cat_labels + dog_labels)

# Reshape images to a flat array
num_samples = len(all_images)
images_flat = all_images.reshape(num_samples, -1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, all_labels, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Perform predictions on the test set
predictions = svm_model.predict(X_test_scaled)

# Evaluate the classifier's performance
print("Accuracy Score:", accuracy_score(y_test, predictions))
print("Detailed Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Function to visualize a few test results
def display_results(images, true_labels, pred_labels, num_results=5):
    plt.figure(figsize=(15, 5))
    for idx in range(num_results):
        plt.subplot(1, num_results, idx + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f'Actual: {true_labels[idx]}, Predicted: {pred_labels[idx]}')
        plt.axis('off')
    plt.show()

# Show a few test images with their predictions
display_results(X_test.reshape(-1, 64, 64), y_test, predictions)