from pipeline.models import RealTimePatchCore
from pipeline.detection import yolo_edge
from pipeline.processors import threaded_gpto1_worker
from pipeline.ui import create_ui_widgets
from pipeline.threads import process_frames_thread, result_thread, gpt_result_updater, video_playback_thread
import torch
import gc
import cv2
import time
import queue
import threading
import io
import traceback
from PIL import Image as PILImage
from ultralytics import YOLO
from IPython.display import display
    
def start_thread(target_func, args):
    t = threading.Thread(target=target_func, args=args)
    t.daemon = True
    t.start()
    return t

def clear_queue(q, name):
    count = 0
    try:
        while not q.empty():
            q.get_nowait()
            count += 1
    except:
        pass
    if count > 0:
        print(f"Cleared {count} items from {name}")

def process_video(
    model_path,
    openai_api_key,  
    video_path=None,
    use_camera=False,
    camera_index=0,
    low_threshold=0.3,
    medium_threshold=0.5,
    high_threshold=0.7,
    object_threshold=0.5,
    detection_area_ratio=0.3,
    system_prompt="How would you describe the anomalies seen on the following paperclip?.",
    min_frames_between_detections=70,
    playback_speed=1.0,
    max_history=5,  
    enable_gpt4=False,
    disable_video_display=False,
    process_every_n_frames=1, 
):
   
    ui_widgets=create_ui_widgets()
    patchcore_model = RealTimePatchCore(model_path=model_path)
    yolo_model = YOLO(r"./yolo11n.pt")
    yolo_model.to("cuda")
    print("models loaded")
    
    class rt_state:
        def __init__(self):
            self.paused = False
            self.shutdown = False
            self.current_speed = playback_speed
            #self.frame_count = 0
            self.total_frames = 0
            
    ad_frame_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue()
    gpt_input_q = queue.Queue(maxsize=5)  
    gpt_output_q = queue.Queue()
    video_frame_queue = queue.Queue(maxsize=100)
    detection_history = []
    detection_display_history = []
    #previous_frame = None 

    print("queues created")
    log_widget = ui_widgets['log_widget']
    video_widget = ui_widgets['video_widget']
    history_widget = ui_widgets['history_widget']
    gpto1_widget = ui_widgets['gpto1_widget']
    stats_widget = ui_widgets['stats_widget']
    output = ui_widgets['output']


    start_thread(process_frames_thread, (
        patchcore_model,
        ad_frame_queue,
        result_queue,
        gpt_input_q,
        enable_gpt4,
        openai_api_key,
        log_widget,
        detection_display_history,
        history_widget,
        low_threshold,
        medium_threshold,
        high_threshold,
        max_history,
    ))
    
    #print("Processing thread started")
    if enable_gpt4 and openai_api_key:
        start_thread(threaded_gpto1_worker, (
            gpt_input_q,
            gpt_output_q,
            openai_api_key,
            system_prompt,
        ))

        start_thread(gpt_result_updater, (
            gpt_output_q,
            detection_display_history,
            history_widget,
        ))
    else:
        with gpto1_widget:
            gpto1_widget.clear_output()
            print("GPT-o1 Analysis disabled")
    
   
            
    start_thread(result_thread, (
        result_queue,
        history_widget,
        detection_display_history,
        max_history,
    ))

    print("result thread started")
    try:
        if use_camera:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                with output:
                    print("could not open camera")
                return 
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                with output:
                    print(f"Error: Could not open video file: {video_path}")
                return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        shared_state = rt_state()
        shared_state.total_frames = total_frames
        shared_state.current_speed = playback_speed 
        
        start_thread(video_playback_thread, (
            cap,
            shared_state,
            video_frame_queue,
        ))
        detected_object_count = 0
        start_time = time.time()
        current_object_threshold = object_threshold
        
        frames_since_last_detection = min_frames_between_detections
        
        # Main video processing loop 
        last_stats_update_time = 0
        stats_update_interval = 1.0
        while not shared_state.shutdown or not video_frame_queue.empty():
            try:
 
                frame_id, frame = video_frame_queue.get(timeout=1.0)
                frame_count = frame_id 
                
                frames_since_last_detection += 1
                can_detect = frames_since_last_detection >= min_frames_between_detections
                should_process = (frame_count % process_every_n_frames == 0)
                progress = frame_count / total_frames if total_frames > 0 else 0

                process_scale = .25
                actual_height, actual_width = frame.shape[:2]
                process_width = int(actual_width * process_scale)
                process_height = int(actual_height * process_scale)
                processing_frame = cv2.resize(frame, (process_width, process_height))
                #print("test 1")

                if not disable_video_display:
                    frame_height, frame_width = processing_frame.shape[:2]  
                    display_scale = 1
                    display_width = int(frame_width * display_scale)
                    display_height = int(frame_height * display_scale)
                    
                    video_display = cv2.resize(processing_frame, (display_width, display_height))
                    
                if can_detect and should_process:
                    try:
                        #print("test")
                        is_centered, confidence, bbox, detection_info = yolo_edge(
                            processing_frame,
                            model=yolo_model,  
                            center_tolerance=detection_area_ratio, 
                            edge_detection_threshold=30,  # For edge detection 
                            min_contour_area=2,    # Allow small contours
                            max_contour_area=15000, # Maximum paperclip size
                            confidence_threshold=current_object_threshold,  # For YOLO detection
                            history_buffer=detection_history,
                            debug_mode=False,
                            use_gpu=False,
                        )
   
                        if not disable_video_display and is_centered and confidence > 0:
                            
                            display_info = detection_info.copy()
                            if 'box_width' in display_info:
                                display_info['box_width'] = int(display_info['box_width'] * display_scale)
                            if 'box_height' in display_info:
                                display_info['box_height'] = int(display_info['box_height'] * display_scale)
                            
                        if is_centered and confidence >= 0.1:
                            with output:
                                method = detection_info.get('method', 'unknown')
                                print(f"PAPERCLIP CENTERED at frame {frame_count} using {method}")
                                print(f"- Confidence: {confidence:.2f}")
                            timestamp = time.time()
                            try:
                                ad_frame_queue.put((frame_count, timestamp, processing_frame.copy()), block=False)
                                detected_object_count += 1
                                frames_since_last_detection = 0
                            except queue.Full:
                                with output:
                                    print("process queue full")
                    except Exception as e:
                        with output:
                            traceback.print_exc()
                            
                if frame_count % 1 == 0 and not disable_video_display:
                    rgb_video = cv2.cvtColor(video_display, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(rgb_video)
                    buf = io.BytesIO()
                    pil_img.save(buf, format='JPEG')
                    video_widget.value = buf.getvalue()
        
                elapsed_time = time.time() - start_time
                process_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                queue_size = ad_frame_queue.qsize()
            
                current_time = time.time()
                if current_time - last_stats_update_time >= stats_update_interval:
                    stats_html = f"""
                    <div style="font-family: monospace; background-color: #2e2e2e; color: #ffffff; padding: 10px; border-radius: 10px;">
                        <div style="font-weight: bold; font-size: 18px;">Status: Processing</div>
                        <div>Progress: {frame_count}/{total_frames} ({progress*100:.1f}%)</div>
                        <div>Playback FPS: {process_fps:.1f}</div>
                        <div>Objects Detected: {detected_object_count}</div>
                        <div>Queue Size: {queue_size}</div>
                        <div>Thresholds: 
                            <span style="color: orange;">Low={low_threshold}</span>, 
                            <span style="color: red;">Medium={medium_threshold}</span>, 
                            <span style="color: darkred;">High={high_threshold}</span>,
                            Object={current_object_threshold}
                        </div>
                        <div>Playback Speed: {shared_state.current_speed}x</div>
                        <div>Showing last {len(detection_display_history)}/{max_history} detections in history</div>
                    </div>
                    """
                    stats_widget.value = stats_html
                    last_stats_update_time = current_time

            except queue.Empty:
                continue    
    except KeyboardInterrupt:
        with output:
            print("Interrupted by user")
    except Exception as e:
        with output:
            print(f"Error: {str(e)}")
            traceback.print_exc()
    finally:
            print("waiting for gpt analysis")
            wait_time = 30 
            if enable_gpt4 and openai_api_key:
                pending = gpt_input_q.qsize() + gpt_output_q.qsize()
                if pending > 0:
                    stats_html = f"""
                    <div style="font-family: monospace; background-color: #2e2e2e; color: #ffffff; padding: 10px; border-radius: 10px;">
                        <div style="font-weight: bold; font-size: 18px;">Status: Waiting for pending analyses</div>
                        <div>Waiting up to {wait_time} seconds for {pending} GPT-4V tasks to complete...</div>
                        <div>Objects Detected: {detected_object_count}</div>
                    </div>
                    """
                    stats_widget.value = stats_html
                    start_wait = time.time()
                    while time.time() - start_wait < wait_time:
                        remaining = gpt_input_q.qsize() + gpt_output_q.qsize()
                        if remaining == 0:
                            break
                        print(f"waiting")
                        time.sleep(1)
                    else:
                        print("timeout")
                        
            shared_state.shutdown = True 
            ad_frame_queue.put(None)
            result_queue.put(None)
            
            if enable_gpt4 and openai_api_key:
                print("telling gpt to worker to stop")
                gpt_input_q.put(None)
                print("waiting for gpt worker to exit")
                time.sleep(3)
                print("telling gpt worker to stop.")
                gpt_output_q.put(None)
                print("waiting for gpt worker to")
                time.sleep(3)


            time.sleep(3)

            clear_queue(ad_frame_queue, "frame queue")
            clear_queue(result_queue, "result queue")
            clear_queue(gpt_input_q, "GPT input queue")
            clear_queue(gpt_output_q, "GPT output queue")
            clear_queue(video_frame_queue, "anomaly detection queue")  
            torch.cuda.empty_cache()
            gc.collect()

            if 'cap' in locals() and cap.isOpened():
                cap.release()
            stats_html = f"""
            <div style="font-family: monospace; background-color: #2e2e2e; color: #ffffff; padding: 10px; border-radius: 10px;">
                <div style="font-weight: bold; font-size: 18px;">Status: Complete</div>
                <div>Total Objects Detected: {detected_object_count}</div>
                <div>Completed successfully.</div>
            </div>
            """
            stats_widget.value = stats_html 
            print("All resources released, shutdown complete")
            
