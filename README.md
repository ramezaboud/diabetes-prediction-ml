<div align="center">

# 🩺 Diabetes Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

**A machine learning project for predicting diabetes using clinical and demographic features.**

[Overview](#overview) •
[Dataset](#dataset) •
[Installation](#installation) •
[Usage](#usage) •
[Project Structure](#project-structure) •
[Contributing](#contributing) •
[License](#license)

</div>

---

## Overview

This repository contains exploratory data analysis (EDA) and machine learning models to predict whether a patient is likely to have diabetes based on various health indicators. The project demonstrates an end-to-end data science workflow including data cleaning, visualization, feature engineering, model training, and evaluation.

## Dataset

| File | Description |
|------|-------------|
| `data/diabetes_dataset__2019.csv` | Primary dataset with clinical measurements (2019) |
| `data/diabetes_prediction_dataset.csv` | Extended dataset for prediction modeling |

### Features

The datasets include features such as:

- **Demographics**: Age, Gender, BMI
- **Medical History**: Hypertension, Heart Disease, Smoking History
- **Clinical Measurements**: Blood Glucose Level, HbA1c Level
- **Target**: Diabetes diagnosis (binary classification)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebooks

Launch Jupyter and open the notebooks:

```bash
jupyter notebook
# or
jupyter lab
```

| Notebook | Purpose |
|----------|---------|
| `notebooks/diabetes_notebook__2019.ipynb` | EDA and analysis of the 2019 dataset |
| `notebooks/diabetes_prediction_notebook.ipynb` | ML model training, evaluation, and prediction |

### Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('data/diabetes_prediction_dataset.csv')

# Prepare features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## Project Structure

```
diabetes-prediction/
├── 📄 README.md                              # Project documentation
├── 📄 LICENSE                                # MIT License
├── 📄 requirements.txt                       # Python dependencies
├── 📄 .gitignore                             # Git ignore rules
├── � data/
│   ├── diabetes_dataset__2019.csv            # Dataset (2019)
│   └── diabetes_prediction_dataset.csv       # Prediction dataset
├── � notebooks/
│   ├── diabetes_notebook__2019.ipynb         # Analysis notebook
│   └── diabetes_prediction_notebook.ipynb    # Prediction notebook
└── 📁 docs/
    ├── Research paper.docx                   # Research paper
    └── GptZero.png                           # Additional documentation
```

## Results

Model performance on the diabetes prediction dataset:

| Model | Accuracy |
|-------|----------|
| **XGBoost** 🏆 | 96.23% |
| Random Forest | 95.64% |
| Decision Tree | 94.51% |
| KNN | 88.84% |
| Logistic Regression | 87.92% |
| SVM | 87.29% |

> Best model: **XGBoost** with 96.23% accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the data science community

</div>