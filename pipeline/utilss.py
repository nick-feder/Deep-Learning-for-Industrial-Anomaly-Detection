import torch
import gc
import cv2
from PIL import Image as PILImage
import base64
import io
import numpy as np

def img_to_base64(img, quality=95):
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    pil_img = PILImage.fromarray(img_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str, (width, height)


def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU Memory] {tag} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB"
              )
        
def preprocess_for_display(frame, resize_dim=256, crop_dim=244):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(frame_rgb)
    width, height = pil_image.size
    if width > height:
        new_width = int(resize_dim * width / height)
        new_height = resize_dim
    else:
        new_width = resize_dim
        new_height = int(resize_dim * height / width)
    pil_resized = pil_image.resize((new_width, new_height), PILImage.BILINEAR)
    left = (new_width - crop_dim) // 2
    top = (new_height - crop_dim) // 2
    right = left + crop_dim
    bottom = top + crop_dim
    pil_cropped = pil_resized.crop((left, top, right, bottom))
    cropped_array = np.array(pil_cropped)
    opencv_image = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2BGR)
    return opencv_image

def create_base64_composite_image(original_frame, overlay):
    if original_frame is None or overlay is None:
        raise ValueError("Missing original frame or overlay")

    if original_frame.shape != overlay.shape:
        raise ValueError("Image shapes do not match")

    combined = np.hstack((original_frame, overlay))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    line_thickness = 2
    text_color = (255, 255, 255)

    h, w = original_frame.shape[:2]
    cv2.rectangle(combined, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.rectangle(combined, (w, 0), (w*2, 30), (0, 0, 0), -1)
    cv2.putText(combined, 'Original Frame', (10, 20), font, font_scale, text_color, line_thickness)
    cv2.putText(combined, 'Anomaly Overlay', (w + 10, 20), font, font_scale, text_color, line_thickness)

    _, buffer = cv2.imencode('.jpg', combined)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image
