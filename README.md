# Human-Activity-Recognization
Deep learning-based human activity recognition using the UCF50 dataset with ConvLSTM. Detect and classify actions effectively. Includes model training and performance evaluation.

# Human Activity Recognition with ConvLSTM

This repository implements a deep learning-based approach to recognize human activities using the UCF50 dataset and ConvLSTM architecture. It includes model training, evaluation, and results analysis.

## Dataset

The project uses the **UCF50** dataset, a widely used video dataset for action recognition, consisting of 50 different action categories.

- Download the dataset from the [official UCF50 website](https://www.crcv.ucf.edu/data/UCF50.php).
- Preprocess the videos into frames for input into the ConvLSTM model.

## Model Architecture

The model leverages:
- **Convolutional Neural Networks (CNNs)** to extract spatial features.
- **Long Short-Term Memory (LSTM)** layers for temporal sequence modeling.

This combination captures both spatial and temporal dynamics of the activities.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Human-Activity-Recognition.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
-   tensorflow==2.10.0
-  numpy==1.22.0
-  pandas==1.4.0
-  scikit-learn==1.0.2
-  matplotlib==3.5.1
-  seaborn==0.11.2
-  opencv-python==4.5.5.64
-  h5py==3.6.0
-  tqdm==4.62.3

## Usage

1. Preprocess the dataset:
   - Organize the UCF50 videos into frames.
   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results

The model achieves high accuracy on the UCF50 dataset, demonstrating its ability to effectively classify human activities. Detailed evaluation metrics and visualizations are available in the results directory.

## File Structure

```
.
├── data/                  # Dataset directory
├── preprocess.py          # Preprocessing script
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── models/                # Model architecture and weights
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! If you have ideas for improving the project or find bugs, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the MIT(LICENSE) file for details.
