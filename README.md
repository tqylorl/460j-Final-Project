# FoodVision: Food Recognition & Calorie Estimation

## Project Overview

FoodVision is an AI-powered food recognition and calorie estimation system that helps users track their nutritional intake through image analysis. The project combines computer vision with nutritional data to provide accurate food identification and calorie information.

### Objectives

- Develop a robust food image classification model using the Food-101 dataset
- Create a system to estimate calorie content based on recognized food items
- Build a user-friendly interface for real-time food recognition and nutritional analysis
- Provide accurate nutritional information to help users make informed dietary choices

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/foodvision.git
   cd foodvision
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the datasets:
   ```bash
   python scripts/download_datasets.py
   ```

## Datasets

### Food-101

The project uses the Food-101 dataset from ETH Zurich, available through HuggingFace. This dataset contains:
- 101,000 food images divided into 101 food categories
- 750 training images and 250 test images per category
- Resolution of approximately 512x512 pixels per image

The dataset can be loaded using the HuggingFace datasets library:
```python
from datasets import load_dataset
ds = load_dataset("ethz/food101")
```

### Calories Dataset

The project includes a custom calories dataset (`calories.csv`) that provides nutritional information for various food items:
- 2,225 food items across multiple categories
- Calorie content per 100 grams
- Energy content in kilojoules (KJ)
- Organized by food category for easy mapping to image classifications

## File Structure

```
foodvision/
├── data/
│   ├── raw/                  # Raw downloaded datasets
│   └── processed/            # Preprocessed data
├── models/
│   ├── saved/                # Saved model checkpoints
│   └── config/               # Model configuration files
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model architecture definitions
│   ├── training/             # Training utilities
│   ├── evaluation/           # Evaluation metrics
│   └── visualization/        # Visualization tools
├── app/
│   ├── streamlit_app.py      # Streamlit web interface
│   └── utils/                # App utilities
├── scripts/
│   ├── download_datasets.py
│   ├── train_model.py
│   └── evaluate_model.py
├── tests/                    # Unit tests
├── requirements.txt          # Project dependencies
├── setup.py                  # Package installation
├── README.md                 # This file
└── LICENSE                   # Project license
```

## Training Models

### Data Preprocessing

Prepare your data for training:
```bash
python scripts/preprocess_data.py --config configs/preprocessing.yaml
```

### Training

Train the food recognition model:
```bash
python scripts/train_model.py --config configs/training.yaml
```

Key training parameters (defined in configs/training.yaml):
- `model_architecture`: ResNet50, EfficientNet, or Vision Transformer
- `batch_size`: 32 (adjust based on available GPU memory)
- `learning_rate`: 0.001
- `num_epochs`: 30
- `weight_decay`: 0.0001
- `data_augmentation`: True (recommended to improve model generalization)

### Fine-tuning

Fine-tune a pre-trained model:
```bash
python scripts/finetune_model.py --model_checkpoint models/saved/best_model.pth --config configs/finetuning.yaml
```

## Evaluation Metrics

The project uses several metrics to evaluate model performance:

- **Top-1 Accuracy**: Percentage of images where the correct food category was predicted with the highest confidence.
- **Top-5 Accuracy**: Percentage of images where the correct food category appears among the top 5 predictions.
- **Precision, Recall, and F1-Score**: Per-class performance metrics.
- **Confusion Matrix**: Visual representation of classification performance.
- **Calorie Estimation Error**: Mean absolute percentage error (MAPE) between predicted and actual calorie values.

Run evaluation:
```bash
python scripts/evaluate_model.py --model_checkpoint models/saved/best_model.pth --test_data data/processed/test
```

## Usage Examples

### Command Line Interface

Predict food type and calories for a single image:
```bash
python scripts/predict.py --image_path path/to/your/food/image.jpg
```

### Web Interface

Launch the Streamlit web app:
```bash
streamlit run app/streamlit_app.py
```

#### Web Interface Functionality

![Upload Interface](docs/images/upload_interface.png)
*Food image upload interface*

![Recognition Results](docs/images/recognition_results.png)
*Food recognition and calorie estimation results*

![Nutritional Breakdown](docs/images/nutritional_breakdown.png)
*Detailed nutritional information*

## Limitations and Future Improvements

### Current Limitations

- The model may struggle with mixed dishes or foods not represented in the training data
- Portion size estimation is approximate and may affect calorie accuracy
- Limited to 101 food categories from the Food-101 dataset
- Lighting and image quality significantly impact recognition accuracy

### Planned Improvements

- Expand the food category coverage to include more cuisines and dishes
- Implement more accurate portion size estimation using depth sensing
- Add detailed macro and micronutrient information
- Develop a mobile application with AR capabilities for real-time recognition
- Incorporate user feedback for continual model improvement
- Add support for recognizing multiple food items in a single image

## References and Acknowledgments

- Food-101 Dataset: Bossard, Lukas, Matthieu Guillaumin, and Luc Van Gool. "Food-101 – Mining Discriminative Components with Random Forests." European Conference on Computer Vision, 2014.
- HuggingFace Datasets: Wolf, Thomas, et al. "HuggingFace's Datasets: Community-driven effort to standardize datasets for machine learning."
- Nutritional data compiled from multiple public databases including USDA Food Data Central.
- Special thanks to the open-source community for tools and libraries that made this project possible.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

