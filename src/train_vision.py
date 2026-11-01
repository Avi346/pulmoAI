"""
train_vision.py — Advanced Chest X-Ray Pneumonia Classifier
Fine-tunes DenseNet-121 with layer-wise discriminative fine-tuning, focal loss,
aspect-preserving padding (384x384), automated lung cropping, and TTA.
"""

import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "chest_xray_densenet121.pth")

# ── Hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
NUM_CLASSES = 2
IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shortcut Learning Prevention: Automated Lung Field Cropping ───────
def crop_lung_region(pil_img):
    """
    Automated lung field cropping using Otsu thresholding + morphology.
    Strips out extraneous hospital monitoring tags & edge markers.
    """
    try:
        img_gray = np.array(pil_img.convert("L"))
        blurred = gaussian_filter(img_gray, sigma=2)
        thresh = threshold_otsu(blurred)
        binary = blurred < thresh
        closed = closing(binary, square(5))

        h, w = closed.shape
        margin_y, margin_x = int(h * 0.1), int(w * 0.1)
        mask_central = closed[margin_y:h - margin_y, margin_x:w - margin_x]

        if not np.any(mask_central):
            return pil_img

        rows = np.any(closed, axis=1)
        cols = np.any(closed, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add 5% padding margin
        pad_h = int((rmax - rmin) * 0.05)
        pad_w = int((cmax - cmin) * 0.05)
        rmin = max(0, rmin - pad_h)
        rmax = min(h, rmax + pad_h)
        cmin = max(0, cmin - pad_w)
        cmax = min(w, cmax + pad_w)

        cropped = pil_img.crop((cmin, rmin, cmax, rmax))
        return cropped
    except Exception:
        return pil_img


# ── Aspect-Ratio Preserving Transform ─────────────────────────────────
class PadToSquare:
    """
    Resize longest edge to size while preserving aspect ratio,
    then pad shorter side with neutral gray (124, 116, 104).
    """
    def __init__(self, size, fill=(124, 116, 104)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        img = crop_lung_region(img)

        w, h = img.size
        if w > h:
            new_w = self.size
            new_h = int(self.size * h / w)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)

        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return TF.pad(img, padding, fill=self.fill)


# ── Data Transforms ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    PadToSquare(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    PadToSquare(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Datasets & Loaders ───────────────────────────────────────────────
def get_dataloaders():
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_set   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=test_transform)
    test_set  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_set.classes


# ── Regularizer: Focal Loss ───────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss targeting subtle ground-glass opacities.
    Penalizes easy/confident classifications.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Model Architecture (DenseNet-121) ─────────────────────────────────
def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze top feature blocks for discriminative fine-tuning
    for name, child in model.features.named_children():
        if name in ["denseblock3", "transition3", "denseblock4", "norm5"]:
            for param in child.parameters():
                param.requires_grad = True

    # Replace classifier head
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, NUM_CLASSES)

    return model.to(DEVICE)


# ── Compute class weights (handles imbalanced dataset) ────────────────
def compute_class_weights(dataset_path):
    dataset = datasets.ImageFolder(dataset_path)
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(DEVICE)


# ── Training loop ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── Validation loop ──────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── Test evaluation with metrics ─────────────────────────────────────
def evaluate(model, loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 50)
    print("TEST RESULTS (DenseNet-121)")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


_cached_vision_model = None
_vision_lock = threading.Lock()

def get_cached_vision_model():
    global _cached_vision_model
    with _vision_lock:
        if _cached_vision_model is None:
            model = build_model()
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            # Warm up PyTorch inference graph
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
                    model(dummy)
            except Exception:
                pass
            _cached_vision_model = model
    return _cached_vision_model

# ── Predict a single image + TTA (used by Streamlit app) ─────────────
def predict(image_path):
    """
    Load trained DenseNet-121 and predict a single X-ray image with TTA.
    Returns: (class_name, confidence)
    """
    model = get_cached_vision_model()

    class_names = ["NORMAL", "PNEUMONIA"]

    image = Image.open(image_path).convert("RGB")

    # Test-Time Augmentation (TTA): original + horizontal flip
    img_orig = test_transform(image).unsqueeze(0).to(DEVICE)
    img_flip = test_transform(image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_orig = torch.softmax(model(img_orig), dim=1)
        out_flip = torch.softmax(model(img_flip), dim=1)
        probs = (out_orig + out_flip) / 2.0
        confidence, predicted = probs.max(1)

    return class_names[predicted.item()], confidence.item()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")

    model = build_model()
    print("DenseNet-121 loaded (denseblock3/4 & classifier unfrozen)")

    class_weights = compute_class_weights(os.path.join(DATA_DIR, "train"))
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    # Separate parameter groups for discriminative learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE_BACKBONE},
        {'params': head_params,     'lr': LEARNING_RATE_HEAD}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\nTraining DenseNet-121 for {NUM_EPOCHS} epochs at {IMAGE_SIZE}x{IMAGE_SIZE}...")
    print("-" * 50)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    evaluate(model, test_loader, class_names)


if __name__ == "__main__":
    main()
