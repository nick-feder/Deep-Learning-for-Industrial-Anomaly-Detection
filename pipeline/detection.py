import cv2
import numpy as np

def yolo_edge(
    frame,
    model=None,
    center_tolerance=0.25,
    edge_detection_threshold=100,
    min_contour_area=10,
    max_contour_area=10000,
    confidence_threshold=0.2,
    history_buffer=None,
    debug_mode=True,
    use_yolo=True,
    use_gpu=False
):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cw, ch = int(w * center_tolerance), int(h * center_tolerance)
    cl, cr = cx - cw // 2, cx + cw // 2
    ct, cb = cy - ch // 2, cy + ch // 2
    frame_area = h * w

    # --- EDGE DETECTION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if use_gpu:
        gmat = cv2.cuda_GpuMat()
        gmat.upload(blur)
        edge_img = cv2.cuda.createCannyEdgeDetector(edge_detection_threshold, edge_detection_threshold * 2).detect(gmat).download()
    else:
        edge_img = cv2.Canny(blur, edge_detection_threshold, edge_detection_threshold * 2)

    dilated = cv2.dilate(edge_img, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    all_detections = []
    min_offset = float('inf')
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_contour_area <= area <= max_contour_area):
            continue
        x, y, w2, h2 = cv2.boundingRect(c)
        if w2 < 100 or h2 < 100:
            continue
        #aspect ratio check
        asp = max(w2, h2) / (min(w2, h2) + 1e-3)
        if not (asp > 1.05 or area > 10):
            continue
        mx, my = x + w2 // 2, y + h2 // 2
        offset = max(abs(mx - cx) / w, abs(my - cy) / h) * 2
        scale = min(1., area / 1000.)
        pos_score = max(0, 1. - offset)
        conf = max(0, min(1, (scale + pos_score) / 2))
        is_centered = cl <= mx <= cr and ct <= my <= cb

        if is_centered and offset < min_offset:
            min_offset = offset
            best = {
                'bbox': (x, y, x + w2, y + h2),
                'confidence': float(conf),
                'area_pixels': int(area),
                'box_width': w2,
                'box_height': h2,
                'center_offset': float(offset),
                'is_centered': True,
                'area_ratio': float(area / frame_area),
                'method': 'edge_detection'
            }

        if debug_mode:
            all_detections.append({
                'bbox': (x, y, x + w2, y + h2),
                'confidence': float(conf),
                'area_pixels': int(area),
                'box_width': w2,
                'box_height': h2,
                'center_offset': float(offset),
                'is_centered': is_centered,
                'aspect_ratio': float(asp)
            })

    if best:
        print("[INFO] Method used: edge_detection")
        if history_buffer:
            history_buffer.append(best)
            if len(history_buffer) > 30:
                history_buffer.pop(0)
        if debug_mode:
            best['all_detections'] = all_detections
            best['edge_image'] = edge_img
        return True, best['confidence'], best['bbox'], best

    # --- YOLO FALLBACK ---
    result = {
        'status': 'no_centered_paperclip',
        'center_tolerance': center_tolerance,
        'method': 'hybrid_failed'
    }
    if debug_mode:
        result['all_detections'] = all_detections
        result['edge_image'] = edge_img

    if use_yolo and model:
        try:
            yolo_result = model.predict(source=frame, verbose=False, device="0")[0]
            boxes = yolo_result.boxes
            best_box = None
            best_conf = 0.

            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                if conf >= confidence_threshold and cl <= mx <= cr and ct <= my <= cb and conf > best_conf:
                    best_box = (x1, y1, x2, y2)
                    best_conf = conf

            if best_box:
                print("[INFO] Method used: yolo")

                return True, best_conf, best_box, {
                    'bbox': best_box,
                    'confidence': best_conf,
                    'method': 'yolo',
                    'is_centered': True
                }

        except Exception as e:
            if debug_mode:
                result['yolo_error'] = str(e)

    return False, 0., (0, 0, 0, 0), result
