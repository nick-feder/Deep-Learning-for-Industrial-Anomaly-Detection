
import torch
import torchvision.transforms as T
import cv2
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class RealTimePatchCore:
    def __init__(
        self,
        model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        resize=256,
        imagesize=244,
        faiss_on_gpu=True,
        faiss_num_workers=0
    ):
        from models.patchcore import patchcore
        from models.patchcore import common
        try:
            from .models.patchcore import utils 
            self.device = utils.set_torch_device([0] if "cuda" in device else [-1])
        except ImportError:
            self.device = torch.device(device)
            if "cuda" in device:
                torch.cuda.set_device(int(device.split(':')[1]) if ':' in device else 0)
        
        self.imagesize = imagesize
        self.transform = T.Compose([
            T.Resize(resize),
            T.CenterCrop(imagesize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)
        self.model = patchcore.PatchCore(self.device)
        self.model.load_from_path(
            load_path=model_path,
            device=self.device,
            nn_method=nn_method
        )
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.1
        self.thickness = 2
        
    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)
        
        tensor_image = self.transform(pil_image)
        return tensor_image.unsqueeze(0).to(self.device) 
        
    def process_frame(self, frame):
        tensor_frame = self.preprocess_frame(frame)
        with torch.no_grad():
            try:
                score, score_map, _, _ = self.model.predict_single(tensor_frame)
                score_map = score_map.reshape(score_map.shape[0], int(np.sqrt(score_map.shape[1])), -1)
                score_map = score_map.cpu().numpy()
                return score[0], score_map[0]
            except (AttributeError, Exception) as e:
                scores_list, masks_list = self.model._predict(tensor_frame)
                anomaly_score = scores_list[0]
                anomaly_mask = masks_list[0]
             
                if isinstance(anomaly_mask, torch.Tensor):
                    anomaly_mask = anomaly_mask.cpu().numpy()
                    
                return anomaly_score, anomaly_mask
    
    def generate_heatmap(self, anomaly_mask, frame_shape, anomaly_score=None, max_expected_score=10, preserve_original_size=False, target_size=(244, 244)):

        if len(anomaly_mask.shape) == 1:
            side_length = int(np.sqrt(anomaly_mask.shape[0]))
            mask_resized = anomaly_mask.reshape(side_length, side_length)
        else:
            mask_resized = anomaly_mask
        mask_min, mask_max = mask_resized.min(), mask_resized.max()
        if mask_max - mask_min > 1e-8:  
            normalized_mask = 255 * (mask_resized - mask_min) / (mask_max - mask_min)
        else:
            normalized_mask = np.zeros_like(mask_resized, dtype=np.float32)
        heatmap = cv2.applyColorMap(
            normalized_mask.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        if preserve_original_size:
            heatmap = cv2.resize(heatmap, target_size)
        else:
            if frame_shape[1] > heatmap.shape[1] * 5 or frame_shape[0] > heatmap.shape[0] * 5:
                heatmap = cv2.resize(heatmap, target_size)
            else:
                heatmap = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]))
        if anomaly_score is not None:
            scale_factor = min(1.0, max(0.3, anomaly_score / max_expected_score))
            heatmap = cv2.multiply(heatmap, scale_factor)
        
        return heatmap
