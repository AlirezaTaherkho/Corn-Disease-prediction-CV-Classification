# Corn Leaf Disease Image Classification

A PyTorch-based image classification project for recognizing four corn leaf conditions:

- **Blight**
- **Common Rust**
- **Gray Leaf Spot**
- **Healthy**

The project compares several training configurations and CNN architectures, evaluates the trained models, and provides a real-time webcam inference script. The **Data Augmentation model** was selected as the best model for the final real-time demonstration.

## Project Overview

The main goal of this project is to classify corn leaf images into four disease/health categories using deep learning.

The project contains four trained model configurations:

1. **Initial Model** — baseline custom CNN.
2. **Balanced Data Model** — custom CNN trained using a more balanced dataset distribution.
3. **Data Augmentation Model** — custom CNN trained with data augmentation.
4. **ResNet50 Model** — transfer learning using a pretrained ResNet50 backbone.

Each model has its own directory containing the trained weights, training history, architecture description, and evaluation results.

## Project Structure

```text
Corn Leaf Diseases/
│
├── balance_data model/
│   ├── architecture.txt
│   ├── balance_data_model.pth
│   ├── classification report & Rank 2 Acc.txt
│   ├── confusion matrix.png
│   ├── data distribution.png
│   ├── model learning plot.png
│   ├── train_accuracy.npy
│   ├── train_loss.npy
│   ├── val_accuracy.npy
│   └── val_loss.npy
│
├── data aug model/
│   ├── architecture.txt
│   ├── data_aug_model.pth
│   ├── classification report & Rank 2 Acc.txt
│   ├── confusion matrix.png
│   ├── data distribution.png
│   ├── model learning plot.png
│   ├── train_accuracy.npy
│   ├── train_loss.npy
│   ├── val_accuracy.npy
│   └── val_loss.npy
│
├── initail model/
│   ├── architecture.txt
│   ├── initial_model.pth
│   ├── classification report & Rank 2 Acc.txt
│   ├── confusion matrix.png
│   ├── data distribution.png
│   ├── model learning plot.png
│   ├── train_accuracy.npy
│   ├── train_loss.npy
│   ├── val_accuracy.npy
│   └── val_loss.npy
│
├── resnet50 model/
│   ├── architecture.txt
│   ├── resnet50_model.pth
│   ├── classification report & Rank 2 Acc.txt
│   ├── confusion matrix.png
│   ├── data distribution.png
│   ├── model learning plot.png
│   ├── train_accuracy.npy
│   ├── train_loss.npy
│   ├── val_accuracy.npy
│   └── val_loss.npy
│
├── evaluation.ipynb
├── model_creation.ipynb
└── result.py
```

> The folder and file names above reflect the current project organization. If you rename them, update the paths used in the notebooks and `result.py`.

## Dataset

The notebooks use a dataset organized with `torchvision.datasets.ImageFolder`.

The expected dataset structure is:

```text
Corn_data/
├── train/
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
│
├── valid/
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
│
└── test/
    ├── Blight/
    ├── Common_Rust/
    ├── Gray_Leaf_Spot/
    └── Healthy/
```

The class names are read directly from the `ImageFolder` dataset.


## Models

### 1. Initial, Balanced Data, and Data Augmentation Models

These three configurations use the same custom CNN architecture.

The network consists of four convolutional blocks followed by three fully connected sections.

```text
Input: 3 × 200 × 200

Conv Block 1
├── Conv2d(3 → 16, kernel=3)
├── BatchNorm2d
├── ReLU
└── MaxPool2d

Conv Block 2
├── Conv2d(16 → 32, kernel=3)
├── BatchNorm2d
├── LeakyReLU
├── MaxPool2d
└── Dropout(0.2)

Conv Block 3
├── Conv2d(32 → 64, kernel=3)
├── BatchNorm2d
├── LeakyReLU
└── MaxPool2d

Conv Block 4
├── Conv2d(64 → 128, kernel=3)
├── BatchNorm2d
├── LeakyReLU
├── MaxPool2d
└── Dropout(0.2)

Classifier
├── Flatten
├── Linear(128 × 10 × 10 → 256)
├── ReLU
├── Linear(256 → 64)
├── ReLU
├── Dropout(0.2)
└── Linear(64 → 4)
```

The final layer produces four logits, one for each class.

The difference between these three experiments is mainly the data/training configuration rather than the CNN architecture:

- **Initial Model:** baseline experiment.
- **Balanced Data Model:** experiment using a more balanced class distribution.
- **Data Augmentation Model:** experiment using image augmentation to increase training variation.

### 2. ResNet50

The ResNet50 experiment uses a pretrained ResNet50 model from `torchvision`.

The pretrained parameters are frozen:

```python
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
```

The original classification layer is replaced with a four-class classifier:

```python
fc_in_features = model.fc.in_features
model.fc = nn.Linear(fc_in_features, 4)
```

This is a transfer-learning approach: the pretrained ResNet50 feature extractor is retained while the final classification layer is adapted to the four corn leaf classes.

## Training

The training process is implemented in `model_creation.ipynb`.

The main training components are:

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** Adam
- **Initial learning rate:** `0.0005`
- **Learning-rate scheduler:** `StepLR`
- **Batch size:** `64`
- **Input size:** `200 × 200`
- **Maximum epochs:** `60`
- **Device:** CPU in the provided notebook
- **Early stopping:** enabled


Training and validation metrics are saved as NumPy arrays:

```text
train_accuracy.npy
train_loss.npy
val_accuracy.npy
val_loss.npy
```

The trained model weights are saved as `.pth` files.

## Evaluation

`evaluation.ipynb` is used to analyze the trained models.

The evaluation includes:

### Training Curves

The saved NumPy arrays are loaded and plotted to visualize:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

### Classification Report

Scikit-learn's `classification_report` is used to calculate class-level classification metrics on the test set.

This provides metrics such as precision, recall, F1-score, and support for each class.

### Confusion Matrix

A confusion matrix is generated using:

It is displayed with the corresponding class names to show which classes are correctly or incorrectly predicted.

### Top-2 / Rank-2 Accuracy

The evaluation also calculates a Rank-2 Accuracy.

For each test image, the model's two highest-scoring classes are selected. A prediction is counted as correct when the true class appears among these two classes.

This metric is useful for measuring whether the correct class is at least among the model's two most likely predictions.

## Saved Results

Each model directory contains the main artifacts produced during training and evaluation.

| File | Purpose |
|---|---|
| `architecture.txt` | Saved model architecture|
| `*.pth` | Trained PyTorch model weights |
| `classification report & Rank 2 Acc.txt` | Classification report and Rank-2 accuracy |
| `confusion matrix.png` | Test-set confusion matrix |
| `data distribution.png` | Dataset class distribution |
| `model learning plot.png` | Training/validation learning curves |
| `train_accuracy.npy` | Training accuracy history |
| `train_loss.npy` | Training loss history |
| `val_accuracy.npy` | Validation accuracy history |
| `val_loss.npy` | Validation loss history |

## Real-Time Inference

`result.py` runs the selected model on a live webcam stream using OpenCV.

## Requirements

The project requires Python and the following main packages:

```text
numpy
matplotlib
opencv-python
torch
torchvision
torchsummary
scikit-learn
```

They can be installed with:

```bash
pip install numpy matplotlib opencv-python torch torchvision torchsummary scikit-learn
```

## How to Run

### 1. Prepare the dataset

Place the dataset in the expected `train`, `valid`, and `test` structure with four class folders in each split.

### 2. Train a model

Open:

```text
model_creation.ipynb
```

Update the dataset and output paths if necessary, then run the notebook.

The notebook trains the model, evaluates training/validation performance during training, restores the best model state using early stopping, and saves the model weights and training histories.

### 3. Evaluate the model

Open:

```text
evaluation.ipynb
```

Update the model and dataset paths, load the desired `.pth` file, and run the notebook to generate the classification report, confusion matrix, learning plots, and Rank-2 accuracy.

### 4. Run real-time classification

Make sure the model path in `result.py` points to the trained Data Augmentation model.

Then run:

```bash
python result.py
```

The webcam window can be closed by pressing **Esc**.

## Notes

- The notebooks currently use absolute Windows paths. These paths must be changed when the project is moved to another computer.
- The model expects **RGB, 200 × 200** input images.
- The inference preprocessing should remain consistent with the preprocessing used during training.
- The provided training code uses CPU (`device = "cpu"`). A CUDA-enabled GPU can be used if the code and environment are configured accordingly.
- The final real-time inference script uses the **Data Augmentation model**, which is identified in the script as the best model.
- Model comparison should be based on the saved test-set metrics and evaluation artifacts rather than training accuracy alone.

## Technologies

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Jupyter Notebook**

## Project Pipeline

```text
Corn Leaf Dataset
        │
        ▼
ImageFolder + Preprocessing
        │
        ├───────────────┬────────────────┬────────────────┐
        ▼               ▼                ▼                ▼
 Initial CNN      Balanced Data     Data Augmentation   ResNet50
        │               │                │                │
        └───────────────┴────────────────┴────────────────┘
                                │
                                ▼
                         Model Evaluation
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Classification  Confusion   Rank-2
                  Report       Matrix     Accuracy
                                │
                                ▼
                     Best Model Selection
                                │
                                ▼
                         Webcam Inference
```
