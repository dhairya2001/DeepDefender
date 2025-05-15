# Real vs. Fake Classifier

This project implements a machine learning pipeline to classify videos as either **real** or **fake**, using the FaceForensics++ dataset. The entire workflow is encapsulated in the Jupyter notebook `Real_Fake_Classifier.ipynb`.

## Project Structure

```
DeepDefender/
├── Real_Fake_Classifier.ipynb  # Main notebook for model training and evaluation
├── FF++/                       # Dataset directory with 'real', 'fake', and 'eval_videos' folders
├── output/                     # Directory for saved model checkpoints and logs
└── README.md                   # Project documentation
```

## Features

- Preprocesses video data from the FaceForensics++ dataset
- Resizes video frames to 224×224 for uniform input
- Extracts frame-level features and trains a custom Multi-Layer Perceptron (MLP) classifier
- Utilizes mixed-precision training for faster computation on GPUs
- Logs evaluation metrics: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
- Implements utilities such as early stopping and learning rate scheduling

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/dhairya2001/DeepDefender.git
   cd DeepDefender
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *(You can generate this file using `pip freeze > requirements.txt` if needed.)*

## Requirements

The project depends on the following Python packages:

- `torch`, `torchvision`, `timm`
- `opencv-python`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `tqdm`

## Model Configuration

All hyperparameters and directory paths can be customized via the `config` dictionary:

```python
config = {
  'real_dir': 'FF++/real',
  'fake_dir': 'FF++/fake',
  'eval_dir': 'FF++/eval_videos/',
  'output_dir': 'output/',
  'resize_width': 224,
  'resize_height': 224,
  'batch_size': 1024,
  'feature_dim': 20,
  'hidden_dims': [512, 256, 128, 128, 64],
  'dropout_rate': 0.5,
  'learning_rate': 0.001,
  'weight_decay': 1e-5,
  'mixed_precision': True,
  'grad_accum_steps': 4,
  'num_epochs': 1000,
  'patience': 500,
  'val_split': 0.5,
  'test_split': 0.1
}
```

## Evaluation Metrics

The following metrics are used to evaluate the model:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

All metrics are visualized using `matplotlib` and `seaborn`.

## Training & Inference

To train or evaluate the model, open the notebook in Jupyter and run all cells:

```bash
jupyter notebook Real_Fake_Classifier.ipynb
```

The best-performing model is saved to the `output/` directory during training.

## Dataset

This project uses the [FaceForensics++ dataset](https://www.kaggle.com/datasets/hungle3401/faceforensics). The dataset should be placed locally in the following structure:

```
FF++/
├── real/
├── fake/
└── eval_videos/
```

Make sure the videos or frames are preprocessed and placed in the appropriate directories before training.
