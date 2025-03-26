import json
import numpy as np


def load_roi(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    # Store ROI polygons
    rois = []
    labels = []
    for roi in data["contours"]:
        vertices = np.array(roi["vertices"], dtype=np.int32)
        rois.append(vertices)
        labels.append(roi["label"])  # Save ROI labels
    return rois, labels

def write_roi(file_path,contours,labels):
    data = {"contours":[]}
    for i in range(len(contours)):
        data["contours"].append({"label":labels[i],"vertices":contours[i]})
    with open(file_path, "w") as file:
        json.dump(data,file,indent=4)
