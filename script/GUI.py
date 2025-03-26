import cv2
import numpy as np
import subprocess
from PyQt6.QtWidgets import (
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QSlider, QCheckBox, QSpinBox, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt
from dataReader import DataReader
from contrastAdjustment import ContrastAdjustment
import roiAdapter as roiA
import roiComputation as roiC
from segmentation import Segmentation
from mplCanvas import MplCanvas
from csvAdapter import write_F_to_csv
from multiselectComboBox import MultiSelectComboBox

class Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Image Segmentation")
        self.setGeometry(100, 100, 900, 500)  # Largeur ajustée pour trois colonnes

        self.data = None

        # Layout principal en horizontal (3 colonnes)
        self.main_layout = QHBoxLayout()

        # ---- Colonne gauche : Boutons ----
        self.buttons_layout = QVBoxLayout()

        self.load_button = QPushButton("Load TIFF Video")
        self.load_button.clicked.connect(self.load_image)
        self.buttons_layout.addWidget(self.load_button)

        self.load_roi_button = QPushButton("Load ROI")
        self.load_roi_button.clicked.connect(self.load_roi)
        self.buttons_layout.addWidget(self.load_roi_button)

        self.metadata_layout = QVBoxLayout()
        self.metadata_labels = {}
        self.buttons_layout.addLayout(self.metadata_layout)

        self.buttons_layout.addStretch()  # Garde les boutons en haut et pousse le reste vers le bas

        self.segment_button = QPushButton("Segmentation")
        self.segment_button.clicked.connect(self.perform_segmentation)
        self.buttons_layout.addWidget(self.segment_button)

        self.modify_roi_button = QPushButton("Modify ROI")  # Nouveau bouton
        self.modify_roi_button.clicked.connect(self.modify_roi)
        self.buttons_layout.addWidget(self.modify_roi_button)

        # ---- Ajout du bouton Reset ----
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_application)
        self.buttons_layout.addWidget(self.reset_button)

        # Ajouter le layout des boutons à la colonne de gauche
        self.main_layout.addLayout(self.buttons_layout)

        # ---- Colonne centrale : Image + Slider + Play/Pause ----
        self.image_layout = QVBoxLayout()

        # Channel selection
        self.channel_combo_box = QComboBox()
        self.channel_selected = 0 # Default to first channel
        self.image_layout.addWidget(self.channel_combo_box)

        # Image
        self.label = QLabel("No file loaded")  # Image
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)  # Slider
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.seek_video)
        self.image_layout.addWidget(self.slider)

        self.image_buttons_layout = QHBoxLayout()

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        
        self.image_buttons_layout.addWidget(self.play_pause_button)

        self.image_select_spinbox = QSpinBox()
        self.image_select_spinbox.editingFinished.connect(lambda: self.image_select_update_v(self.image_select_spinbox.value()))
        self.image_buttons_layout.addWidget(self.image_select_spinbox)

        self.image_layout.addLayout(self.image_buttons_layout)
        # Ajouter l'image et le slider à la colonne centrale
        self.main_layout.addLayout(self.image_layout)

        # ---- Colonne droite : Contraste ----
        self.contrast_layout = QVBoxLayout()
        self.contrast_checkbox = QCheckBox("Apply Contrast Adjustment")
        self.contrast_checkbox.stateChanged.connect(self.update_frame)
        self.contrast_layout.addWidget(self.contrast_checkbox)

        self.contrast_buttons_layout = QHBoxLayout()
        self.contrast_auto_button = QPushButton("Auto contrast")
        self.contrast_auto_button.clicked.connect(self.auto_contrast)
        self.contrast_buttons_layout.addWidget(self.contrast_auto_button)
        self.contrast_min_spinbox = QSpinBox()
        self.contrast_min_spinbox.setRange(0,100000)
        self.contrast_min_spinbox.editingFinished.connect(lambda: self.contrast_min_update_v(self.contrast_min_spinbox.value()))
        self.contrast_buttons_layout.addWidget(self.contrast_min_spinbox)
        self.contrast_max_spinbox = QSpinBox()
        self.contrast_max_spinbox.setRange(0,100000)
        self.contrast_max_spinbox.editingFinished.connect(lambda: self.contrast_max_update_v(self.contrast_max_spinbox.value()))
        self.contrast_buttons_layout.addWidget(self.contrast_max_spinbox)
        self.contrast_layout.addLayout(self.contrast_buttons_layout)

        # Histogram
        self.histogram_plot = MplCanvas(self)
        self.histogram_plot.axes.set_title("Intensity")
        self.contrast_layout.addWidget(self.histogram_plot)

        # dF/F
        self.dff_compute_button = QPushButton("Compute dF/F")
        self.dff_compute_button.clicked.connect(self.compute_dff)
        self.contrast_layout.addWidget(self.dff_compute_button)
        self.dff_plot_buttons_layout = QHBoxLayout()
        self.dff_plot_button = QPushButton("Plot dF/F :")
        self.dff_plot_button.clicked.connect(self.plot_dff)
        self.dff_plot_buttons_layout.addWidget(self.dff_plot_button)
        self.dff_multi_combo_box = MultiSelectComboBox()
        self.dff_plot_buttons_layout.addWidget(self.dff_multi_combo_box)
        self.contrast_layout.addLayout(self.dff_plot_buttons_layout)
        self.dff_plot = MplCanvas(self)
        self.dff_plot.axes.set_title("dF/F")
        self.contrast_layout.addWidget(self.dff_plot)

        # Aligner la checkbox en haut à droite
        self.contrast_layout.addStretch()  # Ajoute un espace flexible en bas

        # Ajouter la colonne contraste à droite
        self.main_layout.addLayout(self.contrast_layout)

        # Appliquer le layout principal
        self.setLayout(self.main_layout)

        self.tiff_file_path = None  # Stocker le chemin du TIFF
        self.roi_file_path = None  # Stocker le chemin du ROI
        self.global_contours = None  # Store contours permanently for the whole video
        self.global_labels = None
        
        self.tiff_loaded = False
        self.roi_loaded = False
        self.current_frame = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)
        self.timer_timing = 100 #Default to 100ms between slices when playing the video
        self.contrast_adjuster = ContrastAdjustment()
        self.contrast_min,self.contrast_max = None,None
        self.segmenter = Segmentation()

    def load_image(self):
        self.label.setText("Loading...")

        file_path, _ = QFileDialog.getOpenFileName(self, "Open a File", "", "TIFF Files (*.tiff *.tif)")
        if file_path:
            self.tiff_file_path = file_path  # Stocker le chemin du TIFF
            if self.data:
                self.data.close()
                self.tiff_loaded = False
            self.data = DataReader(file_path)
            self.tiff_loaded = True

            if self.tiff_loaded:
                for key in self.data.metadata:
                    if isinstance(self.data.metadata[key],str):
                        if key in self.metadata_labels:
                            self.metadata_labels[key].setText(key+": "+self.data.metadata[key])
                        else:
                            self.metadata_labels[key] = QLabel(key+": "+self.data.metadata[key])
                            self.metadata_layout.addWidget(self.metadata_labels[key])
                for key in self.metadata_labels:
                    if not key in self.data.metadata:
                        self.metadata_layout.removeWidget(self.metadata_labels[key])
                self.slider.setMaximum(int(self.data.metadata["SizeT"]) - 1)
                self.current_frame = 0
                self.slider.setValue(self.current_frame)
                self.timer_timing = int(float(self.data.metadata["TimeIncrement"])*1000)
                self.label.setText("")
                self.channel_combo_box.clear()
                for i in range(len(self.data.metadata["Channels"])):
                    self.channel_combo_box.addItem(self.data.metadata["Channels"][i])
                self.channel_combo_box.currentIndexChanged.connect(self.channel_combo_box_update)
                if "UR" in self.data.metadata["Channels"]:
                    self.channel_combo_box.setCurrentText("UR")
                self.contrast_min,self.contrast_max = (int(self.data.get_slice(self.channel_selected,self.current_frame).min()),int(self.data.get_slice(self.channel_selected,self.current_frame).max()))
                self.image_select_spinbox.setRange(0,int(self.data.metadata["SizeT"])-1)
            print(f"TIFF File Loaded: {self.tiff_file_path}")  # Debugging
            self.update_frame()

    def load_roi(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Contours File", "", "ROI Files (*.json *.mescroi *.roi)")
        if file_path:
            self.roi_file_path = file_path  # Stocker le chemin du ROI
            self.global_contours, self.global_labels = roiA.load_roi(file_path)
            self.dff_multi_combo_box.addItems(self.global_labels)
            self.update_frame()
            self.roi_loaded = True
            print(f"ROI File Loaded: {self.roi_file_path}")  # Debugging

    def channel_combo_box_update(self):
        self.channel_selected = self.channel_combo_box.currentIndex()
        self.contrast_checkbox.setChecked(False)
        self.contrast_min,self.contrast_max = (int(self.data.get_slice(self.channel_selected,self.current_frame).min()),int(self.data.get_slice(self.channel_selected,self.current_frame).max()))
        self.update_frame()

    def image_select_update_v(self,value):
        self.current_frame = value
        self.update_frame()

    def auto_contrast(self):
        if self.tiff_loaded:
                _,self.contrast_min,self.contrast_max = self.contrast_adjuster.select_contrast(self.data.get_slice(self.channel_selected,self.current_frame))
        self.contrast_checkbox.setChecked(True)
        self.update_frame()

    def contrast_min_update_v(self,value):
        self.contrast_min = value
        self.update_frame()

    def contrast_max_update_v(self,value):
        self.contrast_max = value
        self.update_frame()

    def contrast_check_update(self,_):
        if self.contrast_checkbox.isChecked():
            if self.tiff_loaded:
                img,self.contrast_min,self.contrast_max = self.contrast_adjuster.select_contrast(self.data.get_slice(self.channel_selected,self.current_frame),new_min=self.contrast_min,new_max=self.contrast_max)
        self.update_frame()

    def plot_histogram(self):
        if self.tiff_loaded:
            image = self.data.get_slice(self.channel_selected,self.current_frame)
            histo, _ = np.histogram(image,int(image.max()))
            self.histogram_plot.axes.cla()
            self.histogram_plot.axes.plot(histo[self.contrast_min:self.contrast_max+1])
            self.histogram_plot.axes.set_title("Intensity")
            self.histogram_plot.draw()
            self.contrast_min_spinbox.setValue(self.contrast_min)
            self.contrast_max_spinbox.setValue(self.contrast_max)

    def compute_dff(self):
        if(self.tiff_loaded and self.roi_loaded):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save intensity over time", "", "CSV file (*.csv)")
            self.dff_compute_button.setText("Compute dF/F (in progress)")
            masks = roiC.contours_to_masks(self.global_contours)
            self.f_data,self.dff_data = roiC.compute_dff(self.data,masks,self.channel_selected)
            write_F_to_csv(file_path,self.global_labels,self.f_data)
            self.dff_compute_button.setText("Compute dF/F (done)")

    def plot_dff(self):
        self.dff_plot.axes.cla()
        for i in self.dff_multi_combo_box.getCurrentIndexes():
            self.dff_plot.axes.plot(self.dff_data[i])
        self.dff_plot.axes.legend(self.dff_multi_combo_box.getCurrentIndexes())
        self.dff_plot.draw()

    def timer_timeout(self):
        # Update the slider position and move to the next frame
        self.current_frame = (self.current_frame + 1) % int(self.data.metadata["SizeT"])
        self.slider.setValue(self.current_frame)
        self.update_frame()

    def update_frame(self):
        if self.tiff_loaded:
            img = self.data.get_slice(self.channel_selected,self.current_frame)
            self.slider.setValue(self.current_frame)
            self.image_select_spinbox.setValue(self.current_frame)
            # Apply contrast adjustment if enabled
            if self.contrast_checkbox.isChecked():
                img, _, _ = self.contrast_adjuster.select_contrast(img,new_min=self.contrast_min,new_max=self.contrast_max)
            self.plot_histogram()

            # Normalize and convert the image to color format
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

            # Always display segmentation contours if they exist
            if self.global_contours:
                for i,contour in enumerate(self.global_contours):
                    points = np.array(contour, dtype=np.int32)
                    cv2.drawContours(img_color, [points], -1, (255, 0, 0), 1)  # Draw contours in red
                    cv2.putText(img_color, self.global_labels[i], contour[0], cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 100), 1, cv2.LINE_AA)

            # Convert to QPixmap and display in the interface
            height, width, _ = img_color.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_color.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)

    # Function to seek to a specific frame
    def seek_video(self):
        self.current_frame = self.slider.value()
        self.update_frame()

    def toggle_play_pause(self):
        if self.is_playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(self.timer_timing)
            self.play_pause_button.setText("Pause")

        self.is_playing = not self.is_playing

    def perform_segmentation(self):
        if self.tiff_loaded:
            img = self.data.get_slice(self.channel_selected,self.current_frame)  # Get the current frame
            self.global_contours = self.segmenter.segment(img)  # Store contours for all frames

            if self.global_contours:
                self.roi_loaded = True
                print(f"Segmentation applied. Contours stored for all frames.")

                # Demander où enregistrer le fichier ROI
                save_path, _ = QFileDialog.getSaveFileName(self, "Save ROI File", "", "ROI Files (*.roi)")
                if save_path:
                    roiA.write_roi(save_path, self.global_contours, [str(i) for i in range(len(self.global_contours))])
                    print(f"ROI file saved at {save_path}")
                    self.roi_file_path = save_path  # Met à jour le chemin du ROI file après l'enregistrement

            self.update_frame()  # Update display

    def modify_roi(self):
        if self.tiff_loaded and self.roi_loaded:
            print(f"Opening ROI Editor for:\n - TIFF: {self.tiff_file_path}\n - ROI: {self.roi_file_path}")
            subprocess.run([
                            "python", "modifROI.py",
                            self.tiff_file_path,
                            self.roi_file_path,
                            str(self.current_frame + self.channel_selected*int(self.data.metadata["SizeT"])),
                            str(self.contrast_min),
                            str(self.contrast_max)
                        ], check=True)
        else:
            print("TIFF or ROI file not loaded yet.")

    def reset_application(self):
        """Resets all paths, contours, and the displayed image."""
        print("Resetting application...")

        self.data.close()
        self.tiff_loaded,self.roi_loaded = False,False
        self.tiff_file_path = None
        self.roi_file_path = None
        self.global_contours = None
        self.global_labels = None
        self.current_frame = 0
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.label.setText("No file loaded")
        self.label.setPixmap(QPixmap())  # Clear the image display
        self.histogram_plot.axes.cla() # Clear histogram plot
        self.dff_plot.axes.cla() # Clear dF/F plot
        self.contrast_checkbox.setChecked(False)
        self.timer.stop()
        self.timer_timing = 100
        self.play_pause_button.setText("Play")
        self.dff_compute_button.setText("Compute dF/F")

        print("Application reset completed.")
