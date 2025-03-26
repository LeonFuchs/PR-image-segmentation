import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

class DataReader:
    def __init__(self, file_path):
        """
        Initialize reader with an OME-TIFF file
        Metadata are read from the first slice only
        """
        self.file_path = file_path
        self.file = tifffile.TiffFile(file_path)
        self.metadata = dict()
        self.get_metadata(mode=True)
        #print(self.metadata)
        
        if int(self.metadata["SizeC"]) > 1:
            self.shape = (int(self.metadata["SizeC"]),int(self.metadata["SizeT"]),int(self.metadata["SizeX"]),int(self.metadata["SizeY"]))
        else:
            self.shape = (int(self.metadata["SizeT"]),int(self.metadata["SizeX"]),int(self.metadata["SizeY"]))
        #print(f"File ready with shape : {self.shape}")

    def close(self):
        self.file.close()

    def get_slice(self, channel=0, z_slice=0):
        """Get specified image from the file"""
        return self.file.pages[channel*int(self.metadata["SizeT"])+z_slice].asarray()
    
    def get_all_slices(self,channel=-1):
        """Get all slices, or all in a single channel

        Args:
            channel (int,optional): Channel to get slices from. -1 by default to get all images in the file
        """
        if(channel==-1):
            return [page.asarray() for page in self.file.pages]
        else:
            return [page.asarray() for page in self.file.pages[channel*int(self.metadata["SizeT"]):(channel+1)*int(self.metadata["SizeT"])]]

    def get_metadata(self,slice=0,keys=["TimeIncrement","PhysicalSizeX","PhysicalSizeY","SizeC","SizeT","SizeX","SizeY","Type","Channels"],mode=False):
        """
        Gets metadata information from the OME-XML block from a slice in the file. All metadata is read as strings.
        Default metadata retrieved are :
            * TimeIncrement : time in seconds between images
            * PhysicalSizeX : width in μm of pixels
            * PhysicalSizeY : height in μm of pixels
            * SizeC : number of channels
            * SizeT : number of images in a channel
            * SizeX : number of pixels horizontally
            * SizeY : number of pixels vertically
            * Type : type of data stored in each pixel
            * Channels : names of channels

        Args:
            slice (int,optional): Index of the slice to load metadata from. Index to use is when all channels are flattened one ofter the other. 0 by default
            keys (list of string,optional): list of keys to parse in OME-XML metadata
            mode (bool,optional) : whether read metadata should be saved in the DataReader instance. True by default
        """
        tags = self.file.pages[slice].tags
        image_desc = tags["ImageDescription"].value
        pixel_info = image_desc[image_desc.find("<Pixels"):image_desc.find("</Pixels>")]
        channel_info = image_desc[image_desc.find("<Channel"):image_desc.find("<TiffData/>")]
        def parse(str,key,pre=' ',end='"'):
            """Parse str for the string after key and before (and including) end"""
            tmp = str[str.find(pre+key+'=')+len(key)+3:]
            tmp = tmp[:tmp.find(end)]
            return tmp
        res = dict()
        #Fill the dictionary with metadata from keys
        for key in keys:
            res[key] = parse(pixel_info,key)
        if "Channels" in keys:
            #Fill channel data with names (usually color codes)
            res["Channels"] = []
            for i in range(int(res["SizeC"])):
                res["Channels"].append(parse(channel_info,"Name"))
                channel_info = channel_info[channel_info.find("/>")+2:]
        if mode:
            self.metadata = dict(res)
        return res
    
    def channel_from_name(self,name):
        """Returns the channel index corresponding to the input channel name, or -1 if it isn't in the file

        Args:
            name (string): name of the channel to search for

        Returns:
            int: index of channel, or -1 if inexistent
        """
        for i in range(len(self.metadata["Channels"])):
            if self.metadata["Channels"][i] == name:
                return i
        return -1

    # def to_dataframe(self, channel=0, z_slice=0):
    #     """Convertit la tranche sélectionnée en DataFrame Pandas."""
    #     image_2d = self.get_slice(channel, z_slice)
    #     return pd.DataFrame(image_2d)

    def plot_image(self, channel=0, z_slice=0):
        """Plot specified image with a color bar"""
        image_2d = self.get_slice(channel, z_slice)
        plt.figure(figsize=(8, 6))
        plt.imshow(image_2d, cmap="gray", aspect="auto")
        plt.colorbar(label="Intesity")
        plt.title(f"Channel {channel}, Slice {z_slice}")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.show()

    def plot_intensity_profile(self, channel=0, z_slice=0):
        """Plot intensity profile on median row of specified slice"""
        image_2d = self.get_slice(channel, z_slice)
        y_row = image_2d.shape[0] // 2  # Ligne médiane
        intensity_profile = image_2d[y_row, :]

        plt.figure(figsize=(8, 4))
        plt.plot(intensity_profile, label=f"Intensity profile - Row {y_row}")
        plt.xlabel("Position X (pixels)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()
        plt.title("Intensity profile on median row")
        plt.show()

    def plot_histogram(self, channel=0, z_slice=0):
        """Plot histogram of specified slice"""
        image_2d = self.get_slice(channel, z_slice)

        plt.figure(figsize=(8, 6))
        plt.hist(image_2d.ravel(), bins=256, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Intensity")
        plt.ylabel("Number of pixels")
        plt.title("Intensity histogram")
        plt.grid(True)
        plt.show()

if __name__ == '__main__' :

    
    file_path = Path(__file__).parent.parent/"TIFFfiles"/"230425_S2_002.ome.ome.tiff"
    reader = DataReader(file_path)
    print(reader.metadata)
'''
    # Afficher une tranche spécifique (Canal 1, Tranche 1)
    reader.plot_image(channel=1, z_slice=1)
    
    # Tracer le profil d’intensité
    reader.plot_intensity_profile(channel=1, z_slice=1)
    
    # Afficher l’histogramme des intensités
    reader.plot_histogram(channel=1, z_slice=1)
'''
