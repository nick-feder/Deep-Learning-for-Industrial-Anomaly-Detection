import time
import queue
import requests
import numpy as np
from pipeline.utilss import  create_base64_composite_image

    
def threaded_gpto1_worker(input_queue, output_queue, api_key, system_prompt):
    print(" GPT worker starting")
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        test_request = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Test message"},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 5
        }
        
        print(" Testing API connection...")
        test_response = requests.post(api_url, headers=headers, json=test_request)
        print(f"Test response: {test_response.status_code}")
        test_response.raise_for_status()
        print("API connection verified")
        
    except Exception as e:
        print(f"API connection error: {str(e)}")
        output_queue.put({
            'frame_id': -1,
            'timestamp': time.time(),
            'status': 'ERROR',
            'gpto1_result': f"API Connection Error: {str(e)}"
        })

    while True:
            try:
                item = input_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if item is None:
                print("Received shutdown signal")
                break
            
            print(f"Processing frame {item.get('frame_id', 'unknown')}")
            
            if not item.get('is_anomaly', False):
                print("Not an anomaly, skipping")
                item['gpto1_result'] = "Not analyzed (normal frame)"
                output_queue.put(item)
                continue
            
            original_frame = item.get('original_frame')        
            overlay = item.get('overlay')                
            
            try:
                base64_image = create_base64_composite_image(original_frame, overlay)
                context_message = "The image shows the original frame on the left and the anomaly heatmap overlay on the right. Red areas indicate detected anomalies."
                print(f"Encoded image ({len(base64_image)} chars)")
            except Exception as e:
                print(f"Image processing error: {str(e)}")
                item['gpto1_result'] = f"Error processing images: {str(e)}"
                output_queue.put(item)
                continue
            
            request_data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Analyze this paperclip anomaly. Anomaly Score: {item['score']:.3f}. {context_message} Please describe the specific defect or anomaly you observe in detail, focusing on what makes this paperclip abnormal or defective."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            try:
                print("Sending request to OpenAI API...")
                start_time = time.time()
                response = requests.post(api_url, headers=headers, json=request_data, timeout=60)
                process_time = time.time() - start_time

                if response.status_code != 200:
                    msg = f"API Error: Status code {response.status_code}"
                    print(msg)
                    item['gpto1_result'] = msg
                    output_queue.put(item)
                    return

                response_data = response.json()
                choices = response_data.get('choices', [])
                if not choices:
                    msg = "Error: Invalid API response format"
                    print(msg)
                    item['gpto1_result'] = msg
                    output_queue.put(item)
                    return

                gpto1_result = choices[0]['message']['content']
                item['gpto1_result'] = gpto1_result
                item['gpto1_process_time'] = process_time
                print(f"Successfully processed frame {item['frame_id']}")
                output_queue.put(item)

            except Exception as e:
                msg = f"Request or response error: {str(e)}"
                print(msg)
                item['gpto1_result'] = msg
                output_queue.put(item)
    print(" Worker shutting down")