from pipeline.utilss import preprocess_for_display
from IPython.display import display, HTML
from pipeline.ui import create_history_html
import cv2
import time
import numpy as np
import queue
import traceback
from PIL import Image as PILImage
import io

def video_playback_thread(video_source, shared_state, ad_frame_q):
    frame_count = 0
    while not shared_state.shutdown:
        if shared_state.paused:
            time.sleep(0.1)
            continue

        has_frame, frame = video_source.read()
        if not has_frame:
            print("No more frames")
            shared_state.shutdown = True
            break
        
        frame = cv2.flip(cv2.transpose(frame), 1)
        frame_count += 1
        try:
            ad_frame_q.put((frame_count, frame), block=False)
        except queue.Full:
            pass

    
def process_frames_thread(model, frame_q, result_q, gpt_q, use_gpt, api_key,
                          log_widget, display_history, history_widget,
                          low_t, med_t, high_t, max_history):
    print("anomaly detection thread started")
    while True:
        try:
            print("test waiting for frame")
            item = frame_q.get(timeout=5.0)
            if item is None:
                with log_widget:
                    print("Processing thread shutting down.")
                break

            frame_id, timestamp, frame = item
            start = time.time()

            print(f"Processing frame {frame_id}")
            score, mask = model.process_frame(frame)

            display_frame = preprocess_for_display(frame, 256, 244)
            heatmap = model.generate_heatmap(mask, (244, 244, 3), score, True, (244, 244))

            if score > high_t:
                status, alpha = "HIGH ANOMALY", 0.9
            elif score > med_t:
                status, alpha = "MEDIUM ANOMALY", 0.7
            elif score > low_t:
                status, alpha = "LOW ANOMALY", 0.5
            else:
                status = "NORMAL"
                heatmap = np.zeros_like(heatmap)
                alpha = 0

            overlay = cv2.addWeighted(cv2.resize(display_frame, (244, 244)),
                                      1 - alpha, cv2.resize(heatmap, (244, 244)),
                                      alpha, 0)

            result = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'score': float(score),
                'is_anomaly': score > low_t,
                'process_time': time.time() - start,
                'heatmap': heatmap,
                'heatmap_display': heatmap.copy(),
                'original_frame': display_frame,
                'raw_frame': frame.copy(),
                'overlay': overlay,
                'status': status
            }

            if result['is_anomaly'] and use_gpt and api_key:
                try:
                    result['detection_id'] = f"frame_{frame_id}"
                    gpt_q.put(result.copy(), block=False)
                except queue.Full:
                    print("GPT input queue full.")

            result_q.put(result)

        except queue.Empty:
            continue
        except Exception:
            traceback.print_exc()
            result_q.put({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'error': 'Processing failed',
                'status': 'ERROR'
            })
        frame_q.task_done()

    
def gpt_result_updater(gpt_output_q, detection_display_history, history_widget):
    while True:
        result = gpt_output_q.get()
        if result is None:
            print("GPT updater got shutdown signal")
            break

        frame_id = result.get('frame_id')
        gpto1_result_text = result.get('gpto1_result', '')

        for i, item in enumerate(detection_display_history):
            if item.get('frame_id') == frame_id:
                item['gpto1_result'] = gpto1_result_text
                with history_widget:
                    history_widget.clear_output(wait=True)
                    display(HTML(create_history_html(detection_display_history)))
                print(f"Updated GPT result for frame {frame_id}")
                break


def result_thread(result_q, history_widget, display_history, max_history):
    while True:
        try:
            result = result_q.get(timeout=1.0)
            if result is None:
                break

            if 'heatmap' in result and 'original_frame' in result:
                heatmap = result['heatmap']
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB) if len(heatmap.shape) == 2 else heatmap
                img = PILImage.fromarray(cv2.resize(heatmap_rgb, (244, 244)))
                img.save(io.BytesIO(), format='PNG')

                display_history.append(result)
                if len(display_history) > max_history:
                    display_history.pop(0)

                with history_widget:
                    history_widget.clear_output()
                    display(HTML(create_history_html(display_history)))

        except queue.Empty:
            time.sleep(0.01)
