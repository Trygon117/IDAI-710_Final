import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_disease_confusion_matrix(true_labels, predictions, class_names):
    # Calculate the raw grid of numbers
    cm = confusion_matrix(true_labels, predictions)
    
    # Create a visual heat map so it is easy to read
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.title('Disease Confusion Matrix')
    plt.show()

def plot_soft_confusion_matrix(true_labels, predicted_probs, class_names, title='Soft Probability Heatmap'):
    # Create an empty 6x6 grid
    num_classes = len(class_names)
    soft_cm = np.zeros((num_classes, num_classes))
    
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    # Average the predicted probabilities for each true class
    for i in range(num_classes):
        # Find all patients who actually have disease i
        class_indices = np.where(true_labels == i)[0]
        
        if len(class_indices) > 0:
            # Grab their soft probability predictions
            class_probs = predicted_probs[class_indices]
            
            # Calculate the average probability assigned to each of the 6 diseases
            soft_cm[i, :] = np.mean(class_probs, axis=0)

    # Create a custom colormap
    brand_teal = "#009384"
    custom_cmap = sns.light_palette(brand_teal, as_cmap=True)

    # Create a static visual heat map
    plt.figure(figsize=(10, 8))
    sns.heatmap(soft_cm, annot=True, fmt='.1%', cmap=custom_cmap, 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Disease')
    plt.xlabel('Average Predicted Probability')
    plt.title(title)
    plt.show()