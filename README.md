# Skin Disease Detection

This project uses deep learning (TensorFlow/Keras, TF-Hub, and Gradio) for skin disease (benign/malignant) image classification and interactive model demonstration.

## Features

- **Google Colab integration**: Loads data from Google Drive.
- **Model**: MobileNetV2 feature extraction, fine-tunable.
- **Training pipeline**: Data augmentation, validation split.
- **Visualization**: Training/validation accuracy and loss.
- **Gradio demo**: Web-based prediction via image upload.

## Directory Structure

```
Skin-Diesease-Detection/
├── src/                 # Source code for training and model
├── data/                # Example data or placeholder for datasets
├── notebooks/           # Jupyter/Colab notebooks
├── tests/               # Unit tests and test data
├── requirements.txt     # Python dependencies
├── .gitignore
├── LICENSE
├── README.md
```

## Setup & Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Place training images in a directory like:  
   `data/Training/Dataset/{class_name}/*.jpg`  
   Make sure you have folders for each class (e.g., `Benign`, `Malignant`).

3. **Run Model/Training**  
   - You may use the provided Colab notebook or the scripts in `src/`.
   - Adjust paths if running outside Colab.

4. **Launch Gradio Demo**  
   ```bash
   python src/app.py
   ```
   Or via Colab cell using `gr.Interface`.

## Example Colab

See `notebooks/Skin_Disease_Training.ipynb` for a step-by-step guide.


## License

MIT License (see LICENSE file).