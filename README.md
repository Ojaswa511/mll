# My Machine Learning Projects

This repository contains my ML learning journey.

## Projects

### 1. Digit Classification
- **File**: `train_model.py`
- **Dataset**: Sklearn digits dataset
- **Algorithm**: Logistic Regression
- **Accuracy**: ~95%

### 2. Housing Price Prediction
- **Folder**: `project2-housing/`
- **File**: `housing_prediction.py`
- **Dataset**: California Housing
- **Algorithms**: Linear Regression vs Random Forest
- **Best R2 Score**: Check output

### 3. Text Classification
- **Folder**: `project3-text/`
- **File**: `text_classifier.py`
- **Dataset**: 20 Newsgroups
- **Algorithms**: Naive Bayes vs Logistic Regression
- **Task**: Classify religious vs atheist posts

## How to Run

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run any project:
```bash
python train_model.py
python project2-housing/housing_prediction.py
python project3-text/text_classifier.py
```

## What I Learned

- Setting up Python development environment
- Training ML models locally
- Classification vs Regression
- Working with different data types (images, numbers, text)
- Comparing multiple algorithms
- Saving models with pickle
- Git and GitHub basics