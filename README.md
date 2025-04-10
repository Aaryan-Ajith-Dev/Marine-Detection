# Faster R-CNN for Small Object Detection in Marine Drone Imagery

This project implements and fine-tunes a Faster R-CNN model using PyTorch's `torchvision` library for detecting small marine life in aerial drone images.

The model was developed as part of an industry project focused on automated monitoring of marine environments. The primary challenge tackled was the detection of small, low-contrast marine species in complex underwater or ocean surface backgrounds.

---

## Features
- Fine-tuned Faster R-CNN with customized anchor settings for small object detection.
- Weighted sampling to handle class imbalance.
- Custom IoU computation for evaluation.
- Visualizations of detections using `PIL` and `matplotlib`.
- Clean training loop using PyTorch best practices.
- Anomaly Detection pipeline using PaDiM (for outlier region identification).

---

## Architecture
- Backbone: ResNet50
- Detection Head: Custom FastRCNNPredictor
- RPN: Custom Anchor Generator optimized for small objects
- Dataset Format: COCO-style annotations

---

## Usage

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib pillow
```

### 2. Training
```python
model = get_model(num_classes)
train(model, data_loader, device, num_epochs=20, lr=1e-4)
```

---

## Sample Results
> *Note: Visualizations below are from internal datasets and not shared due to confidentiality.*

(If permitted by company, you can insert blurred or symbolic outputs here)

---

## Dataset
Dataset used in this project belongs to the organization and is not publicly available.

To replicate this work, public datasets like the NOAA Fisheries Dataset or AI for Earth Wildlife Detection datasets can be adapted.

---

## Future Work
- Hyperparameter tuning for better small object detection.
- Experimenting with RetinaNet / EfficientDet.
- Deployment-ready inference pipeline.
- Integrating active learning for faster dataset curation.

---

## Disclaimer
This repository contains code independently developed for object detection tasks. The dataset and any associated outputs belong to the company and are not included here.

---

## License
MIT License

