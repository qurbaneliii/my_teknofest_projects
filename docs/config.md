# Configuration Guide

Configuration is validated via pydantic. Fields (with defaults):

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| confidence_threshold | float | 0.7 | Min score to keep detection |
| model_path | str | models/best.pt | YOLOv8 weights path |
| target_class | str | mouse | Target class name |
| track_distance_threshold | float | 120.0 | Max pixel distance to link detections |
| max_track_age | int | 30 | Frames to keep inactive tracks |
| alert_cooldown | float | 5.0 | Seconds between alerts |
| enable_console_alerts | bool | true | Print alerts to stdout |
| enable_file_alerts | bool | true | Append alerts to file |
| enable_log_alerts | bool | true | Log alerts via logger |
| save_detections | bool | true | Persist JSON summary |
| output_dir | str | detections | Folder for outputs |
| video_output | bool | true | Save annotated video when path given |
| show_preview | bool | true | OpenCV preview window |
| dataset_dir | str? | sample_images | Dataset root for preprocessing |
| preprocess_output_dir | str | outputs | Manifest destination |
| preview_width | int? | null | Resize preview width if set |

Validation is invoked automatically when loading configs or can be called via `config.validate()`.
