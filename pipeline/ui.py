import ipywidgets as widgets
from IPython.display import display
from pipeline.utilss import img_to_base64
def create_ui_widgets():
    log_widget = widgets.Output(layout={
        'border': '1px solid #ccc', 
        'padding': '10px',
        'max_height': '200px', 
        'overflow_y': 'auto',
        'width': '320px'  
    })

    video_widget = widgets.Image(format='jpeg', width=320)

    history_widget = widgets.Output()
    results_widget = widgets.Output()
    gpto1_widget = widgets.Output()

    stats_widget = widgets.HTML(
        value="Initializing...",
        layout=widgets.Layout(
            background_color='#2e2e2e',
            border='1px solid #444',
            padding='10px',
            border_radius='10px',
            width='100%'
        )
    )

    left_column = widgets.VBox([
        widgets.HTML("<h3>Video Feed</h3>"),
        video_widget,
        widgets.HTML("<h3>Log Output</h3>"),
        log_widget
    ])

    right_column = widgets.VBox([
        widgets.HTML("<h3>Detection History</h3>"),
        history_widget
    ])

    layout = widgets.HBox([left_column, right_column])
    display(stats_widget, layout, results_widget)

    return {
        'log_widget': log_widget,
        'video_widget': video_widget,
        'history_widget': history_widget,
        'results_widget': results_widget,
        'gpto1_widget': gpto1_widget,
        'stats_widget': stats_widget,
        'layout': layout,
        'output': widgets.Output(),
    }

def create_history_html(history_items, enable_gpt4=False,openai_api_key=None):

        history_styles = """
        <style>
            .history-item {
                margin-bottom: 15px;
                padding: 15px;
                border-radius: 20px;  /* Increased roundness */
                background-color: #ffffff;
                color: #000000;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                overflow: hidden;  /* Ensures inner content respects the rounded borders */
            }
            .history-item.anomaly-low {
                border: 2px solid rgba(255, 149, 0, 0.4);
            }
            .history-item.anomaly-medium {
                border: 2px solid rgba(255, 59, 48, 0.4);
            }
            .history-item.anomaly-high {
                border: 2px solid rgba(211, 47, 47, 0.4);
            }
            .history-item.normal {
                border: 2px solid rgba(52, 199, 89, 0.4);
            }
            .flex-row {
                display: flex;
                flex-direction: row;
                align-items: center;
            }
            .flex-item {
                flex: 1;
                text-align: center;
                padding: 5px;
            }
            .header {
                text-align: center;
                margin-bottom: 10px;
                font-weight: bold;
                font-size: 16px;
            }
            .score-anomaly-low {
                color: rgba(255, 149, 0, 0.8);
            }
            .score-anomaly-medium {
                color: rgba(255, 59, 48, 0.8);
            }
            .score-anomaly-high {
                color: rgba(211, 47, 47, 0.8);
            }
            .score-normal {
                color: rgba(52, 199, 89, 0.8);
            }
            .gpto1-analysis {
                margin-top: 10px;
                padding: 10px;
                background-color: #f2f2f7;
                border-radius: 12px;
                text-align: left;
                font-size: 12px;
                border-left: 4px solid rgba(0, 122, 255, 0.5);
                color: #333333;
            }
        </style>
        """

        
        html_content = history_styles
        
        for idx, hist_item in enumerate(reversed(history_items)):
            frame_base64, frame_dims = img_to_base64(hist_item['original_frame'])
            heatmap_base64, heatmap_dims = img_to_base64(hist_item['heatmap_display'])
            overlay_base64, overlay_dims = img_to_base64(hist_item['overlay'])
            if hist_item['status'] == 'HIGH ANOMALY':
                border_color = "darkred" 
                border_style = "solid"
            elif hist_item['status'] == "MEDIUM ANOMALY":
                border_color = "red"
                border_style = "solid"
            elif hist_item['status'] == 'LOW ANOMALY':
                border_color = "orange"
                border_style = "solid"
            else:
                border_color = "green"
                border_style = "solid"
            
            gpto1_result_html = ""
            if 'gpto1_result' in hist_item and hist_item['gpto1_result']:
                gpto1_result = hist_item['gpto1_result'].replace("\n", "<br>")
                gpto1_result_html = f"""
                <div class="gpto1-analysis">
                    <strong>GPT-o1 Analysis:</strong><br>
                    {gpto1_result}
                </div>
                """
            elif hist_item['is_anomaly'] and enable_gpt4 and openai_api_key:
                gpto1_result_html = f"""
                <div class="gpto1-analysis" style="color: #666;">
                    <em>Waiting for GPT-o1 Analysis for frame {hist_item['frame_id']}...</em>
                </div>
                """
            
            html_content += f"""
            <div style='margin: 10px; border: 3px {border_style} {border_color}; padding: 5px;'>
                <div style='text-align: center;'>
                    <strong>Frame {hist_item['frame_id']}</strong> - Score: {hist_item['score']:.3f}
                    <span style='color: {border_color}; font-weight: bold;'>
                        {hist_item['status']}
                    </span>
                </div>
                <div style='display: flex; flex-direction: row; justify-content: center;'>
                    <div style='margin-right: 5px; text-align: center;'>
                        <img src='data:image/jpeg;base64,{frame_base64}' 
                            width="{frame_dims[0]}" height="{frame_dims[1]}"
                            style='image-rendering: auto;' title='Original Frame'>
                        <div style='text-align: center; font-size: 10px;'>Original</div>
                    </div>
                    <div style='margin-right: 5px; text-align: center;'>
                        <img src='data:image/png;base64,{heatmap_base64}' 
                            width="{heatmap_dims[0]}" height="{heatmap_dims[1]}"
                            style='image-rendering: auto;' title='Heatmap'>
                        <div style='font-size: 10px;'>Heatmap</div>
                    </div>
                    <div style='text-align: center;'>
                        <img src='data:image/jpeg;base64,{overlay_base64}' 
                            width="{overlay_dims[0]}" height="{overlay_dims[1]}"
                            style='image-rendering: auto;' title='Overlay'>
                        <div style='text-align: center; font-size: 10px;'>Overlay</div>
                    </div>
                </div>
                {gpto1_result_html}
            </div>
            """
        
        return html_content