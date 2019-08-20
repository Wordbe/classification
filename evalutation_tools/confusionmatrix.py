import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(preds, labels, num_classes):
    num_test = len(preds)
    cm = np.zeros((num_classes, num_classes))
    for i in range(num_test):
        cm[labels[i]][preds[i]] += 1
    return cm

def draw_confuision_matrix(confusion_matrix):
    LABELS = ['abnormal', 'normal'] # Replace this with your labels
    plt.figure(figsize = (7,5))
    sn.set(font_scale=1.4)
    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16},
              xticklabels=LABELS,
              yticklabels=LABELS)
    plt.xlabel("Predicted")