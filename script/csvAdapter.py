import numpy as np
import csv

def write_F_to_csv(filepath,labels,f_array):
    with open(filepath,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step"]+labels)
        w_array = f_array.transpose()
        for i in range(w_array.shape[0]):
            writer.writerow([i]+[float(value) for value in w_array[i]])