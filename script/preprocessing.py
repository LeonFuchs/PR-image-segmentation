import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from contrastAdjustment import ContrastAdjustment

class Preprocessor:
    def __init__(self, target_size=(360, 360)):
        """Initialise le préprocesseur avec un ajustement de contraste."""
        self.target_size = target_size
        self.contrast_adjuster = ContrastAdjustment()

    def preprocess_image(self, image_2d, debug=True):
        """
        Applique le prétraitement :
        - Ajustement du contraste
        - Filtrage Gaussien (réduction de bruit)
        - Normalisation (0,1)
        - Redimensionnement
        - Génération de masque adaptatif
        """

        # Ajustement du contraste
        image_adjusted, new_min, new_max = self.contrast_adjuster.select_contrast(image_2d)
        # if debug:
        #     self._plot_step(image_adjusted, f"Contrast Adjusted (Min: {new_min}, Max: {new_max})")

        # Filtrage Gaussien
        image_filtered = gaussian_filter(image_adjusted, sigma=0.4)
        # if debug:
        #     self._plot_step(image_filtered, "After Gaussian Filter")

        # Normalisation
        image_normalized = (image_filtered - np.min(image_filtered)) / (np.max(image_filtered) - np.min(image_filtered) + 1e-8)

        # Redimensionnement
        image_resized = resize(image_normalized, self.target_size, mode='constant', anti_aliasing=True)
        # if debug:
        #     self._plot_step(image_resized, "After Resizing")

        # Calcul du seuil adaptatif (99e percentile)
        threshold = np.percentile(image_resized, 98)
        print(f"Threshold utilisé: {threshold}")

        # Génération du masque
        mask = image_resized > threshold
        # if debug:
        #     self._plot_step(mask.astype(float), "Dynamic Thresholding Mask")

        return image_resized, mask

    def _plot_step(self, image, title):
        """Affiche chaque étape du traitement."""
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.colorbar()
        plt.show()