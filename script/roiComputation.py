import numpy as np
import cv2

def contours_to_masks(contours):
    """Returns the array of masks corresponding to input contours
        Each mask is a tuple of its top left coordinates and a binary array

    Args:
        contours (3d array of int): array of contours
    """
    masks = []
    for contour in contours:
        #Search for maximum coordinates
        minX,maxX,minY,maxY = 0,0,0,0
        for point in contour:
            if point[0] < minX : minX = point[0]
            if point[0] > maxX : maxX = point[0]
            if point[1] < minY : minY = point[1]
            if point[1] > maxY : maxY = point[1]
        #Create the mask
        mask = np.full((maxX-minX+1,maxY-minY+1),False)
        #Fill the mask
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i][j] = (cv2.pointPolygonTest(contour,(i,j),False) >= 0)
        masks.append(((minX,minY),mask))
    return masks

def compute_dff(reader,masks,channel=0):
    """Computes dF/F0 for ROIs defined by their masks, static over the video

    Args:
        reader (dataReader): dataReader from which to pull images from
        masks (list of mask): Masks delimiting ROIs
        channel (int,optional): Channel in the file to pull images from. Defaults to the first one
    
    Returns:
        array of float: 2d array (number of roi, time) of F
        array of float: 2d array (number of roi, time) of dF/F0
    """
    roi_cnt,slice_cnt = len(masks),int(reader.metadata["SizeT"])
    dff = np.zeros((roi_cnt,slice_cnt),float)
    f0 = np.zeros(roi_cnt,float)
    for t in range(slice_cnt):
        print(t)
        image = reader.get_slice(channel,t)
        for k in range(roi_cnt):
            f = 0.
            sizeX,sizeY = np.shape(masks[k][1])
            for i in range(sizeX):
                for j in range(sizeY):
                    if masks[k][1][i][j]: f += image[masks[k][0][0]+i][masks[k][0][1]+j]
            dff[k][t] = f
            f0[k] += f
    #Replace F with (F-F0)/F0
    f_array = dff.copy()
    for k in range(roi_cnt):
        f0[k] /= slice_cnt
        for t in range(slice_cnt):
            dff[k][t] = (dff[k][t]-f0[k])/f0[k]
    return f_array,dff