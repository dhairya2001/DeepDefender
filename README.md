# DeepDefender

DeepDefender is a comprehensive deep learning-based system designed to detect and classify videos as either **real** or **fake**. Leveraging the FaceForensics++ dataset, this project implements a custom Multi-Layer Perceptron (MLP) architecture to analyze frame-level features and determine the authenticity of video content. Additionally, it includes a user-friendly Streamlit web application for real-time inference.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Model Configuration](#model-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Training & Inference](#training--inference)
- [Streamlit Web Application](#streamlit-web-application)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Preprocesses video data from the FaceForensics++ dataset.
- Resizes video frames to 224×224 for uniform input.
- Extracts frame-level features and trains a custom MLP classifier.
- Utilizes mixed-precision training for faster computation on GPUs.
- Logs evaluation metrics: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- Implements utilities such as early stopping and learning rate scheduling.
- Provides a Streamlit web application for real-time video classification.

## Project Structure

```
DeepDefender/
├── Real_Fake_Classifier.ipynb  # Jupyter notebook for model training and evaluation
├── train.py                    # Script for training the model
├── prediction.py               # Script for making predictions
├── Deepdefender.py             # Streamlit web application
├── model/                      # Directory containing model architecture and weights
├── FF++/                       # Dataset directory with 'real', 'fake', and 'eval_videos' folders
├── output/                     # Directory for saved model checkpoints and logs
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dhairya2001/DeepDefender.git
   cd DeepDefender
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *(You can generate this file using `pip freeze > requirements.txt` if needed.)*

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

To train or evaluate the model, you can use the provided Jupyter notebook or Python scripts:

**Using Jupyter Notebook:**

```bash
jupyter notebook Real_Fake_Classifier.ipynb
```

**Using Python Scripts:**

- **Training:**

  ```bash
  python train.py
  ```

- **Prediction:**

  ```bash
  python prediction.py --input <path_to_video>
  ```

The best-performing model is saved to the `output/` directory during training.

## Streamlit Web Application

The project includes a Streamlit web application for real-time video classification.

**To run the application:**

```bash
streamlit run Deepdefender.py
```

This will launch a local web server where you can upload videos and receive classification results in real-time.

## Dataset

This project uses the [FaceForensics++ dataset](https://www.kaggle.com/datasets/hungle3401/faceforensics). The dataset should be placed locally in the following structure:

```
FF++/
├── real/
├── fake/
```

Ensure that the videos or frames are preprocessed and placed in the appropriate directories before training.

## Acknowledgements

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- PyTorch, timm, and the broader open-source machine learning community
