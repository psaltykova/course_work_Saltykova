# course_work_Saltykova
# Vehicle Detection Under Varying Illumination Conditions

This project develops a methodology for detecting vehicles (cars, trucks, and buses) 
in real-world conditions under varying illumination — daytime and nighttime — 
using the YOLOv11s object detection model.

## Project Overview

Three model configurations were trained and evaluated:
- **Daytime model** — trained on daytime images
- **Nighttime model** — trained on nighttime images
- **Mixed model** — trained on combined daytime and nighttime images

Key findings:
- The mixed model with backbone freezing (freeze=10) achieved the best overall performance
- Specialized models degrade significantly when tested outside their training domain
- The mixed model is the most practical choice for real-world deployment
- Optimal confidence thresholds: 0.487 for daytime, 0.442 for nighttime

## Datasets

All images were self-collected on city streets under real-world conditions 
using a mobile phone camera.

- [Daytime Dataset](https://app.roboflow.com/polinas-workspace/day_dataset/2)
- [Nighttime Dataset 1](https://app.roboflow.com/polinas-workspace/night_dataset_1)
- [Nighttime Dataset 2](https://app.roboflow.com/polinas-workspace/dataset_night)

## Model Weights

Trained model weights are available on Google Drive:  
[Download Weights](https://drive.google.com/drive/folders/1bsEcM6QWxl3mFMr1OyopLUfNnmzA3taP?usp=drive_link)

## Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv11s |
| Input resolution | 1280 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | AdamW |
| Pre-trained weights | COCO |

## Requirements

```bash
pip install ultralytics roboflow
```

## Usage

```python
from ultralytics import YOLO

model = YOLO("mixed_freeze.pt")

# Use conf=0.487 for daytime, conf=0.442 for nighttime
results = model.predict("your_image.jpg", conf=0.442)
results[0].show()
```

## Notebook

The main experiment notebook: `course_work.ipynb`
