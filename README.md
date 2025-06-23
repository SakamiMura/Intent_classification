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

- **TF-IDF Model**: Robust baseline with interpretable features
- **GloVe Model**: Semantic understanding through pre-trained embeddings
- **Detailed Analysis**: Performance breakdown by intent categories

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

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
