import json
import cv2
import numpy as np
import tifffile
import sys
import os

if len(sys.argv) < 6:
    print("Usage: python roiHighlightDelete.py <TIFF_FILE_PATH> <ROI_FILE_PATH> <FRAME_INDEX> <CONTRAST_MIN> <CONTRAST_MAX>")
    sys.exit(1)

image_path = sys.argv[1]
json_path = sys.argv[2]
frame_index = int(sys.argv[3])
contrast_min = int(sys.argv[4])
contrast_max = int(sys.argv[5])

if not os.path.exists(image_path) or not os.path.exists(json_path):
    print("Error: TIFF or ROI file not found.")
    sys.exit(1)

with tifffile.TiffFile(image_path) as tif:
    image = tif.pages[frame_index].asarray()
    image = np.clip(image, contrast_min, contrast_max)
    image = ((image - contrast_min) / (contrast_max - contrast_min) * 255).astype(np.uint8)

image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

with open(json_path, "r") as f:
    data = json.load(f)

if "contours" not in data:
    print("Error: No 'contours' key found in ROI file.")
    sys.exit(1)

rois = []
roi_info = []

for roi in data["contours"]:
    vertices = np.array(roi["vertices"], dtype=np.int32)
    rois.append(vertices)
    roi_info.append(roi)

roi_count = len(roi_info)  # compteur de nombres de ROIs 

selected_roi_index = None
mouse_x, mouse_y = -1, -1
undo_stack = []
drawing_mode = False
new_roi = []

zoom_mode = False
zoom_active = False
zoom_rect = None
zoom_x1, zoom_y1, zoom_x2, zoom_y2 = 0, 0, image.shape[1], image.shape[0]

info_width = 350
info_display = np.ones((image.shape[0], info_width, 3), dtype=np.uint8) * 50

# Create a window for instructions
instruction_window = np.zeros((500, 500, 3), dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
instructions = [
    "Instructions:",
    "- Right click: Remove the ROI",
    "- 'B': Back - undo removed ROI",
    "- 'D': Draw Mode",
    "- Left click: Add a point", 
    "- ENTER : Validate the polygon",
    "- 'B': Back - Remove the last point",
    "- 'Z': Zoom Mode",
    "- 'R': Reset Zoom",
    "- 'S': Save the final ROI",
    "- 'ESC': Quit",
    " WARNING : PRESS S TO SAVE"
]
y_offset = 20
for instruction in instructions:
    cv2.putText(instruction_window, instruction, (10, y_offset), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += 20
cv2.imshow("Instructions", instruction_window)

def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def mouse_callback(event, x, y, flags, param):
    global selected_roi_index, mouse_x, mouse_y, drawing_mode, new_roi
    global zoom_mode, zoom_active, zoom_rect, zoom_x1, zoom_y1, zoom_x2, zoom_y2

    if zoom_active:
        x = int(zoom_x1 + (x / image.shape[1]) * (zoom_x2 - zoom_x1))
        y = int(zoom_y1 + (y / image.shape[0]) * (zoom_y2 - zoom_y1))

    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        mouse_x, mouse_y = x, y
    else:
        mouse_x, mouse_y = -1, -1

    if zoom_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            zoom_rect = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and zoom_rect:
            zoom_rect[2], zoom_rect[3] = x, y
            update_display(draw_zoom_rect=True)
        elif event == cv2.EVENT_LBUTTONUP:
            zoom_mode = False
            zoom_active = True
            zoom_x1, zoom_x2 = sorted([zoom_rect[0], zoom_rect[2]])
            zoom_y1, zoom_y2 = sorted([zoom_rect[1], zoom_rect[3]])
            update_display()
        return

    selected_roi_index = None
    if not drawing_mode:
        for i, poly in enumerate(rois):
            if is_inside_polygon((x, y), poly):
                selected_roi_index = i
                break

    if event == cv2.EVENT_RBUTTONDOWN:
        if drawing_mode and new_roi:
            new_roi.pop()
            if not new_roi:
                drawing_mode = False
        elif selected_roi_index is not None:
            undo_stack.append((rois[selected_roi_index], roi_info[selected_roi_index]))
            del rois[selected_roi_index]
            del roi_info[selected_roi_index]

        update_display()

    elif event == cv2.EVENT_LBUTTONDOWN and drawing_mode:
        new_roi.append((x, y))
        update_display()
        

def undo_last_action():
    global new_roi, drawing_mode
    if drawing_mode and new_roi:
        new_roi.pop()
        if not new_roi:
            drawing_mode = False
    elif undo_stack:
        last_roi, last_info = undo_stack.pop()
        rois.append(last_roi)
        roi_info.append(last_info)
    update_display()

def update_display(draw_zoom_rect=False):
    global zoom_active, zoom_x1, zoom_y1, zoom_x2, zoom_y2

    image_display = image.copy()

    if zoom_active:
        image_display = image[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
        image_display = cv2.resize(image_display, (image.shape[1], image.shape[0]))

        scale_x = (zoom_x2 - zoom_x1) / image.shape[1]
        scale_y = (zoom_y2 - zoom_y1) / image.shape[0]

        zoomed_rois = [np.array(
            [[int((p[0] - zoom_x1) / scale_x), int((p[1] - zoom_y1) / scale_y)] for p in roi],
            dtype=np.int32
        ) for roi in rois]
    else:
        zoomed_rois = rois

    for i, poly in enumerate(zoomed_rois):
        color = (0, 255, 255) if i == selected_roi_index else (0, 0, 255)
        thickness = 2 if i == selected_roi_index else 1
        cv2.polylines(image_display, [poly], isClosed=True, color=color, thickness=thickness)

        # Affichage du label à côté du premier point du ROI
        if len(poly) > 0:
            label = roi_info[i].get("label", f"ROI_{i+1}")
            text_pos = tuple(poly[0])
            cv2.putText(image_display, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 100, 0), 1, cv2.LINE_AA)


    if drawing_mode and new_roi:
        for (x, y) in new_roi:
            if zoom_active:
                x = int((x - zoom_x1) / scale_x)
                y = int((y - zoom_y1) / scale_y)
            cv2.circle(image_display, (x, y), 3, (0, 0, 255), -1)

    if draw_zoom_rect and zoom_rect:
        # Adapter rectangle à la vue actuelle si zoom déjà actif
        if zoom_active:
            scale_x = (zoom_x2 - zoom_x1) / image.shape[1]
            scale_y = (zoom_y2 - zoom_y1) / image.shape[0]
            x1 = int((zoom_rect[0] - zoom_x1) / scale_x)
            y1 = int((zoom_rect[1] - zoom_y1) / scale_y)
            x2 = int((zoom_rect[2] - zoom_x1) / scale_x)
            y2 = int((zoom_rect[3] - zoom_y1) / scale_y)
        else:
            x1, y1, x2, y2 = zoom_rect

        cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    combined_display = np.hstack((image_display, info_display))
    # Mise à jour du panneau info_display avec les coordonnées souris
    info_display[:] = 50  # fond gris
    if 0 <= mouse_x < image.shape[1] and 0 <= mouse_y < image.shape[0]:
        text = f"Mouse: ({mouse_x}, {mouse_y})"
        cv2.putText(info_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.imshow("ROIs modification", combined_display)

def save_rois():
    modified_data = {"contours": []}
    for roi, metadata in zip(rois, roi_info):
        new_roi = {
            "label": metadata.get("label", "N/A"),
            "vertices": roi.tolist()
        }
        modified_data["contours"].append(new_roi)

    with open(json_path, "w") as f:
        json.dump(modified_data, f, indent=4)

    print(f"Updated ROI file saved: {json_path}")

cv2.namedWindow("ROIs modification")
cv2.setMouseCallback("ROIs modification", mouse_callback)

while True:
    update_display()
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or cv2.getWindowProperty("ROIs modification", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('b'):
        undo_last_action()
    elif key == ord('s'):
        save_rois()
    elif key == ord('z'):
        zoom_mode = not zoom_mode
        print("Zoom mode:", "ON" if drawing_mode else "OFF")
    elif key == ord('r'):
        zoom_active = False
        zoom_x1, zoom_y1, zoom_x2, zoom_y2 = 0, 0, image.shape[1], image.shape[0]
        update_display()
    elif key == ord('d'):
        drawing_mode = not drawing_mode
        new_roi = []
        print("Draw mode:", "ON" if drawing_mode else "OFF")

    elif key == 13 and drawing_mode and len(new_roi) > 2:
        drawing_mode = False
        roi_count += 1
        roi_label = f"{roi_count}"
        rois.append(np.array(new_roi, dtype=np.int32))
        roi_info.append({"label": roi_label, "vertices": new_roi})
        new_roi = []
        update_display()
        
cv2.destroyAllWindows()
