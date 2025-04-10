import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from collections import Counter
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms


def compute_iou(boxA, boxB):
    """
    Compute IoU between two sets of bounding boxes.
    boxA: (N, 4) tensor of ground truth boxes
    boxB: (M, 4) tensor of predicted boxes
    Returns: IoU matrix of shape (N, M)
    """
    boxA = boxA.unsqueeze(1)  # Shape (N, 1, 4)
    boxB = boxB.unsqueeze(0)  # Shape (1, M, 4)

    xA = torch.max(boxA[..., 0], boxB[..., 0])
    yA = torch.max(boxA[..., 1], boxB[..., 1])
    xB = torch.min(boxA[..., 2], boxB[..., 2])
    yB = torch.min(boxA[..., 3], boxB[..., 3])

    interArea = (xB - xA).clamp(0) * (yB - yA).clamp(0)

    boxAArea = (boxA[..., 2] - boxA[..., 0]) * (boxA[..., 3] - boxA[..., 1])
    boxBArea = (boxB[..., 2] - boxB[..., 0]) * (boxB[..., 3] - boxB[..., 1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

# --------------
# Training loop
# ----------------
def train(model, data_loader, device, num_epochs=20, lr=0.0001):
    # Set model to training mode and move to device
    model.train()
    model.to(device)
    
    # Define optimizer (using the lr argument)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            # print(targets)
            targets = [
                {
                    "boxes": torch.tensor(t[0]["bbox"], dtype=torch.float32, device=device).view(-1, 4),
                    "labels": torch.tensor([t[0]["category_id"]], dtype=torch.int64, device=device),
                    "image_id": torch.tensor([t[0]["image_id"]], dtype=torch.int64, device=device),
                    "area": torch.tensor([t[0]["area"]], dtype=torch.float32, device=device),
                    "iscrowd": torch.tensor([t[0]["iscrowd"]], dtype=torch.int64, device=device),
                }
                for t in targets if len(t) > 0 # and "bbox" in t[0] and len(t[0]["bbox"]) > 0

            ]
            # print("Should print False", any(t["boxes"].numel() == 0 for t in targets))  # Should print False


            # Forward pass: model returns a dict of losses during training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")
    return model

# Saving
def save(model, name="fasterrcnn_custom.pth"):
    torch.save(model.state_dict(), name)

def load(model, name="fasterrcnn_custom.pth"):
    # Load model later
    model.load_state_dict(torch.load(name))
    model.eval()

# ------
# Inference
# ---------
def test(test_data_loader, model, device, score_threshold=0.6, iou_threshold=0.3):
    model.to(device)
    model.eval()
    
    ious = []
    for images, targets in test_data_loader:
        images = [img.to(device) for img in images]
        targets = [t[0] for t in targets if len(t) > 0]
        
        with torch.no_grad():
            predictions = model(images)

            
        for img, pred, target in zip(images, predictions, targets):
            print("Raw Predictions:", pred)  # DEBUG: Check predictions
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            print("Filtered predictions:", pred_boxes.shape[0])  # How many are left?

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]

            print("After Score Filtering:", pred_boxes.shape[0])
            image = transforms.ToPILImage()(img.cpu())
            draw = ImageDraw.Draw(image)
            
            gt_boxes = torch.tensor(target["bbox"]).view(-1, 4)
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            
            if len(pred_boxes) > 0:
                keep_indices = nms(pred_boxes, pred_scores[keep], iou_threshold)
                pred_boxes = pred_boxes[keep_indices]
                
                iou_values = compute_iou(gt_boxes, pred_boxes).max(dim=1).values.tolist()
                ious.extend(iou_values)
                
                for box in pred_boxes:
                    x1, y1, x2, y2 = box.tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            for box in gt_boxes:
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
    
    if ious:
        avg_iou = sum(ious) / len(ious)
        print(f"Average IoU: {avg_iou:.4f}")
    else:
        print("No predictions met the threshold.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define smaller anchor sizes
    anchor_generator = AnchorGenerator(
        sizes=((2,), (4,), (8,), (16,), (32,)),  
        aspect_ratios=((0.5, 1.0, 2.0),) * 5 
    )

    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace default RPN with custom anchor generator
    model.rpn.anchor_generator = anchor_generator
    
    model.rpn.pre_nms_top_n_train = 12000
    model.rpn.pre_nms_top_n_test = 300
    model.rpn.post_nms_top_n_train = 6000
    model.rpn.post_nms_top_n_test = 100

    model.roi_heads.box_min_size = 1  # Default is usually ~16


    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2  # Increase output_size if needed
    )

    # Replace the pre-trained head with a new one (for your dataset)
    num_classes = 2  # For example: 1 class (object) + background = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Freeze first two blocks (low/mid features)
    for param in model.backbone.body.layer1.parameters():
        param.requires_grad = False
    for param in model.backbone.body.layer2.parameters():
        param.requires_grad = False
    for param in model.backbone.body.layer3.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = False 

    # Fine-tune high-level features
    for param in model.backbone.body.layer4.parameters():
        param.requires_grad = True

    # Unfreeze the RoI Head (Classification + Bounding Box Regression)
    for param in model.roi_heads.parameters():
        param.requires_grad = True  

    # Define transforms for images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # Paths to your dataset
    root = r"F:\left-cam-img"          # Folder with images (can include subfolders)
    annFile = r"F:\coco_dataset\annotations.json"  # COCO annotation file

    # Load dataset
    true_dataset = CocoDetection(root=root, annFile=annFile, transform=transform)
    dataset = true_dataset
    # # testing
    # test_size = 10  # Number of test samples
    # test_indices = list(range(test_size))  # First `test_size` samples
    # test_indices.append(1211)
    # test_indices.append(1218)

    # dataset = Subset(true_dataset, test_indices)
    # for _, target in dataset:
    #     print(target)


    # --- Handle class imbalance with Weighted Random Sampling ---
    # Create weights for each sample:
    #   For example, assign weight 1.0 if the image has at least one annotation,
    #   and weight 0.5 if it has no annotations (background only).
        
    # Step 1: Count occurrences of each class
    counts = Counter(len(target) == 0 or all(t["category_id"] == 0 for t in target) for _, target in dataset)  
    # counts[False]  -> Number of images with objects
    # counts[True] -> Number of background-only images

    # Step 2: Compute weights as inverse frequency
    total = sum(counts.values())
    print(counts.items())
    class_weights = {key: total / count for key, count in counts.items()}  
    print(class_weights)

    weights = [
        class_weights[len(target) == 0 or all(t["category_id"] == 0 for t in target)]
        for _, target in dataset
    ]

    weights = torch.DoubleTensor(weights)
    print("Weights", weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Define DataLoader using the sampler instead of shuffle=True
    data_loader = DataLoader(dataset, batch_size=16, sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))

    # Train the model
    model = train(model, data_loader, device, num_epochs=10, lr=1e-8)
    
    # save the model
    save(model)
    # load(model)
    # Testing the model
    test_dataset = Subset(true_dataset, [1211, 1218])
    for _, target in test_dataset:
        print(target)
    test_data_loader = DataLoader(test_dataset, batch_size=16,  collate_fn=lambda x: tuple(zip(*x)))
    # Test on a sample image
    test(test_data_loader, model, device)

if __name__ == "__main__":
    main()
