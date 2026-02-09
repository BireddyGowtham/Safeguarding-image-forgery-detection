# Image Forgery Detection using Swin Transformer

This project implements an image forgery detection system using the Swin Transformer architecture with PyTorch. The model is trained on the CASIA datasets, which include both authentic and tampered images.

## Project Structure

```
image-forgery-swin-pytorch
├── src
│   ├── train.py          # Main entry point for training the model
│   ├── st.py             # Complete training script for image forgery detection
│   ├── dataset.py        # Dataset loading and preprocessing
│   ├── model.py          # Swin Transformer model definition
│   ├── evaluate.py       # Model evaluation functions
│   └── utils.py          # Utility functions for reproducibility and dataset downloading
├── configs
│   └── default.yaml      # Configuration settings for hyperparameters and dataset paths
├── data
│   ├── CASIA1           # CASIA v1 dataset (authentic and tampered images)
│   └── CASIA2           # CASIA v2 dataset (authentic and tampered images)
├── models
│   └── best_model.pth    # Best model weights after training
├── notebooks
│   └── exploration.ipynb  # Jupyter notebook for exploratory data analysis
├── scripts
│   └── run_train.sh       # Shell script to automate the training process
├── requirements.txt       # Required Python packages and their versions
├── .gitignore             # Files and directories to ignore by Git
└── README.md              # Project documentation
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd image-forgery-swin-pytorch
pip install -r requirements.txt
```

## Dataset

The CASIA datasets need to be downloaded manually. After downloading, extract them into the `data` directory as follows:

```
data/
├── CASIA1/   # CASIA v1 dataset
└── CASIA2/   # CASIA v2 dataset
```

## Training the Model

To train the model, run the following command:

```bash
bash scripts/run_train.sh
```

This will execute the training script with the necessary configurations.

## Evaluation

After training, the best model weights will be saved in the `models` directory. You can evaluate the model using the evaluation functions defined in `src/evaluate.py`.

## Usage

You can use the trained model for inference on new images by loading the model weights from `models/best_model.pth` and passing images through the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.