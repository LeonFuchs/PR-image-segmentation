import numpy as np

class ContrastAdjustment:
    def __init__(self):
        pass

    def select_contrast(self,image,criteria=10000,new_min=-1,new_max=-1):
        """Selects a smaller contrast band. The selection criteria is TODO to be refined.

        Args:
            image (numpy.ndarray): image to process
            criteria (int,optional): intensity selection criteria. Defaults to 10000. TODO to be refined
            new_min (int, optional): forces the lower bound of the contrast band. Defaults to -1 to let it be chosen according to the automated criteria
            new_max (int, optional): _description_. forces the upper bound of the contrast band. Defaults to -1 to let it be chosen by the according to the automated criteria

        Returns:
            numpy.ndarray: image with new contrast
            int : minimum intensity of new image
            int : maximum intensity of new image
        """
        if new_min == -1 or new_max == -1:
            origin_min,origin_max = image.min(),image.max()
            histogram,bin_edges = np.histogram(image,int(origin_max))
            criteria = image.size/criteria
            
            if new_min == -1:
                new_min = 0
                for i in range(new_min,new_max):
                    if histogram[i]>criteria:
                        new_min=i
                        break

            if new_max == -1:
                new_max = histogram.size-1
                for i in range(new_max,new_min,-1):
                    if histogram[i]>criteria:
                        new_max=i
                        break
        
        new_image = np.copy(image)
        for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                if new_image[i][j] > new_max:
                    new_image[i][j] = new_max
                if new_image[i][j] < new_min:
                    new_image[i][j] = new_min
                new_image[i][j] = min(new_max+1,image[i][j]-new_min)
                

        return new_image,new_min,new_max