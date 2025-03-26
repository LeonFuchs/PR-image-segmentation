import numpy as np
import cv2
from tensorflow.keras.models import load_model

class Segmentation:
    def __init__(self):
        """Loads the U-Net segmentation model"""
        self.model = load_model("script\model_unet.keras")
        print(" Model successfully loaded")

    def segment(self, img):
        """
        Performs image segmentation and processes detected contours.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            list: List of processed contours.
        """
        if img is None:
            print(" No image provided for segmentation.")
            return []

        # Resize and normalize the image
        original_size = (img.shape[1], img.shape[0])  # Store original size
        img_resized = cv2.resize(img, (360, 360))  # Adjust to model input size
        img_resized = img_resized.astype(np.float32) / 255.0  # Normalize
        img_resized = np.expand_dims(img_resized, axis=[0, -1])  # Format (1, 360, 360, 1)

        # Predict the segmentation mask
        mask_pred = self.model.predict(img_resized)[0]
        mask_pred = (mask_pred > 0.5).astype(np.uint8)  # Apply binary threshold
        mask_pred = cv2.resize(mask_pred, original_size)  # Resize back to original dimensions

        # Normalize and binarize the predicted mask
        gray = cv2.normalize(mask_pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Detect contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        min_contour_area = 50
        max_contour_area = 250
        processed_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > max_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)

                if w > h:
                    cnt_left = cnt[cnt[:, :, 0] < x + w // 2]
                    cnt_right = cnt[cnt[:, :, 0] >= x + w // 2]
                    processed_contours.append(cnt_left)
                    processed_contours.append(cnt_right)
                else:
                    cnt_top = cnt[cnt[:, :, 1] < y + h // 2]
                    cnt_bottom = cnt[cnt[:, :, 1] >= y + h // 2]
                    processed_contours.append(cnt_top)
                    processed_contours.append(cnt_bottom)
            elif area > min_contour_area:
                processed_contours.append(cnt)

        roi_contours = [cnt[:, 0, :].tolist() if cnt.ndim == 3 else cnt.tolist() for cnt in processed_contours if cnt is not None]

        return roi_contours