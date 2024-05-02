import librosa  # For audio signal computations as MFCC
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import dct

import csv

from classification.utils.plots import show_confusion_matrix
from classification.datasets import Dataset, get_cls_from_path
from classification.utils.plots import plot_audio, plot_specgram


dataset_size = 50

classes = ['birds','chainsaw','fire','handsaw','helicopter']
classnames = ['birds','chainsaw','fire','handsaw','helicopter'] #, "garbage"] #to have the garbage class in matrix

ground_class = []

filename = "ground_classes.csv"

file = open(filename, 'w')
file.close()
class_value = 0
for classe in classes:
    for i in range(int(dataset_size/5)):

        ground_class.append(classe)

        file = open(filename, "a")
        file.write(f"{class_value}\n")
        #file.write(f"{classe}\n")
        file.close()


with open('predicted_class.csv', newline='') as csvfile:
    # Create a CSV reader object
    csv_reader = csv.reader(csvfile, delimiter='\n', quotechar="'")
    
    # Initialize an empty list
    lst = []
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        value = row[0].strip("[]'")
        if (value == "birds"):
            lst.append(0)
        if (value == "chainsaw"):
            lst.append(1)
        if (value == "fire"):
            lst.append(2)
        if (value == "handsaw"):
            lst.append(3)
        if (value == "helicopter"):
            lst.append(4)
        if (value == "garbage"):
            lst.append(5)        

# Print the resulting list
print(lst[0])
print(ground_class[0])

lst = np.asarray(lst)
ground_class = np.repeat([0,1,2,3,4],int(dataset_size/5)) #np.asarray(ground_class)
print(lst)
print(ground_class)

acc = np.sum(lst == ground_class) / len(lst)

print('Accuracy of CNN with fixed train/validation sets : {:.1f}%'.format(100*acc))
show_confusion_matrix(lst, ground_class, classes)