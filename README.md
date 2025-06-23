# Intent Classification Project

A machine learning system for classifying user intents using the CLINC150 dataset. This project implements both TF-IDF and GloVe-based text vectorization approaches with Logistic Regression for intent recognition.

## 🎯 Overview

This project builds a robust intent classification system capable of understanding and categorizing user queries across 150 different intent classes. The system is designed for practical applications like chatbots, virtual assistants, and customer service automation.

## ✨ Features

- **Dual Vectorization Methods**: TF-IDF and GloVe embeddings for text representation
- **Comprehensive Preprocessing**: Text cleaning and normalization pipeline
- **Advanced Analytics**: Detailed performance metrics and visualization tools
- **CLINC150 Dataset**: Industry-standard benchmark with 150 intent classes
- **Modular Architecture**: Clean, maintainable code structure
- **Model Persistence**: Save and load trained models

## 📁 Project Structure

```
Intent_classification/
├── data/
│   ├── data_full.json          # CLINC150 dataset
│   └── glove.6B.50d.txt        # GloVe embeddings (excluded from git)
├── src/
│   ├── load_data.py            # Dataset loading utilities
│   ├── preprocessing.py        # Text preprocessing functions
│   ├── vectorizer.py           # TF-IDF vectorization
│   ├── glove_vectorizer.py     # GloVe embedding utilities
│   ├── train.py                # Model training pipeline
│   └── visualize_results.py    # Results analysis and visualization
├── models/                     # Saved models directory
├── notebooks/                  # Experimental notebooks
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Intent_classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download GloVe embeddings** (optional)
   ```bash
   # Download glove.6B.50d.txt and place in data/ directory
   # Available at: https://nlp.stanford.edu/projects/glove/
   ```

### Usage

**Run the complete pipeline:**
```bash
python main.py
```

**Key Components:**

- **Data Loading**: Automatically processes CLINC150 dataset splits
- **Text Preprocessing**: Cleans and normalizes input text
- **Model Training**: Trains both TF-IDF and GloVe-based models
- **Evaluation**: Comprehensive performance analysis with detailed metrics

## 🔧 Technical Details

### Dataset
- **CLINC150**: 150 intent classes across 10 domains
- **Splits**: Training, validation, test, and out-of-scope samples
- **Domains**: Banking, travel, utility, work, etc.

### Models
- **Base Classifier**: Logistic Regression with L2 regularization
- **TF-IDF Features**: Unigrams and bigrams, max 5000 features
- **GloVe Features**: 50-dimensional word embeddings averaged per sentence

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrix visualization
- Detailed classification reports

## 📊 Results

The system achieves competitive performance on the CLINC150 benchmark:

### TF-IDF Model Performance
- **Test Accuracy**: 89.0%
- **Macro F1-Score**: 0.890
- **Weighted F1-Score**: 0.890
- **Class Distribution**: 78 excellent (F1 ≥ 0.9), 47 good (0.8-0.9), 24 fair (0.7-0.8)
- **Top Performers**: confirm_reservation, flip_coin, roll_dice, routing, traffic (F1 = 1.000)

### GloVe Model Performance  
- **Test Accuracy**: 81.0%
- **Macro F1-Score**: 0.810
- **Weighted F1-Score**: 0.810
- **Characteristics**: Better semantic understanding but lower overall accuracy
- **Best Classes**: tire_pressure, reset_settings, calories (F1 ≥ 0.95)

### Model Comparison
| Metric | TF-IDF | GloVe |
|--------|---------|--------|
| Accuracy | 89.0% | 81.0% |
| Macro F1 | 0.890 | 0.810 |
| Classes F1 ≥ 0.9 | 52.0% | 35.3% |
| Performance | Higher precision | Better semantics |

**Key Insights**: TF-IDF achieves superior overall performance with higher accuracy and F1-scores, while GloVe provides better semantic understanding for complex queries.

## 🛠️ Development

### Adding New Features
1. Create new modules in `src/` directory
2. Follow existing code structure and documentation patterns
3. Update `main.py` to integrate new functionality

### Extending Vectorization
- Implement new vectorizers following the pattern in `glove_vectorizer.py`
- Add corresponding training pipelines in `train.py`

## 📋 Dependencies

- **Core ML**: scikit-learn, numpy, pandas
- **NLP**: nltk, spacy
- **Utilities**: joblib (model persistence)
- **Analysis**: matplotlib, seaborn (for visualization)

## 🎓 Use Cases

- **Chatbot Intent Recognition**
- **Voice Assistant Command Classification**
- **Customer Service Query Routing**
- **Natural Language Interface Development**

