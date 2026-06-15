# 💰 Salary Estimation Using K-Nearest Neighbors (KNN) — Machine Learning

A supervised machine learning project that predicts whether an individual's annual income exceeds **$50,000** based on demographic and employment features. The model uses the **K-Nearest Neighbors (KNN)** classification algorithm trained on the **Adult Census Income** dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Model Architecture](#-model-architecture--knn-classifier)
- [Project Workflow](#-project-workflow)
- [Results & Performance](#-results--performance)
- [Installation & Usage](#-installation--usage)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

| Detail              | Description                                                    |
|---------------------|----------------------------------------------------------------|
| **Project Name**    | Salary Estimation Using KNN — Machine Learning                 |
| **Objective**       | Predict whether an employee earns more or less than $50K/year  |
| **Algorithm**       | K-Nearest Neighbors (KNN) Classifier                          |
| **Dataset**         | Adult Census Income Dataset (UCI Machine Learning Repository)  |
| **Problem Type**    | Binary Classification                                         |
| **Accuracy**        | **81.51%**                                                     |
| **Language**        | Python 3                                                       |

The goal of this project is to build a **binary classifier** that can predict salary class based on four key features extracted from census data. This has practical applications in:

- **HR Analytics** — Estimating expected salary brackets for new hires
- **Financial Services** — Assessing creditworthiness or loan eligibility
- **Policy Research** — Studying income determinants across demographics

---

## 📊 Dataset Description

The dataset used is a subset of the **Adult Census Income Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult), originally extracted from the 1994 U.S. Census database.

### Dataset Statistics

| Property            | Value         |
|---------------------|---------------|
| **Total Records**   | 32,561        |
| **Features**        | 4             |
| **Target Variable** | 1 (income)    |
| **Missing Values**  | None          |

### Features

| Feature              | Type     | Description                          | Min   | Max    | Mean     |
|----------------------|----------|--------------------------------------|-------|--------|----------|
| `age`                | Integer  | Age of the individual                | 17    | 90     | 38.58    |
| `education.num`      | Integer  | Number of years of education         | 1     | 16     | 10.08    |
| `capital.gain`       | Integer  | Capital gains recorded               | 0     | 99,999 | 1,077.65 |
| `hours.per.week`     | Integer  | Average working hours per week       | 1     | 99     | 40.44    |

### Target Variable — `income`

| Class     | Label | Count   | Percentage |
|-----------|-------|---------|------------|
| `<=50K`   | 0     | 24,720  | 75.92%     |
| `>50K`    | 1     | 7,841   | 24.08%     |

> **Note:** The dataset is **imbalanced** — approximately 76% of individuals earn ≤$50K, while only 24% earn >$50K.

---

## 🤖 Model Architecture — KNN Classifier

### What is K-Nearest Neighbors (KNN)?

**K-Nearest Neighbors** is a **non-parametric, instance-based (lazy learning)** supervised classification algorithm. Instead of learning an explicit model during training, KNN stores the entire training dataset and makes predictions by finding the K closest data points (neighbors) to a new input and assigning the most common class label among those neighbors.

### How KNN Works

```
1. Store all training data points in feature space
2. For a new input, calculate the distance to ALL training points
3. Select the K nearest neighbors (closest points)
4. Assign the majority class label among those K neighbors
```

### Model Configuration

| Hyperparameter      | Value           | Description                                              |
|---------------------|-----------------|----------------------------------------------------------|
| `n_neighbors`       | **8**           | Number of nearest neighbors to consider                  |
| `metric`            | **Minkowski**   | Distance metric used for neighbor calculation            |
| `p`                 | **2**           | Power parameter — p=2 corresponds to Euclidean distance  |

### Distance Formula — Minkowski Distance (p=2 → Euclidean)

The Euclidean distance between two points **A** and **B** in n-dimensional space:

$$d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$$

Where:
- $A_i$ and $B_i$ are the feature values of points A and B
- $n$ is the number of features (4 in our case)

### Why K = 8?

The optimal K value was determined by **iterating K from 1 to 39** and plotting the mean error rate for each value. K=8 was selected as it provides a good balance between:

- **Low bias** (not too large a K, which would over-smooth decision boundaries)
- **Low variance** (not too small a K, which would overfit to noisy data)

---

## 🔄 Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA LOADING                                                │
│     └── Load salary.csv using Pandas                            │
│                                                                 │
│  2. DATA PREPROCESSING                                          │
│     ├── Encode target variable (<=50K → 0, >50K → 1)           │
│     ├── Separate features (X) and target (Y)                    │
│     └── Train-Test Split (80% train / 20% test)                 │
│                                                                 │
│  3. FEATURE SCALING                                             │
│     └── StandardScaler (Z-score normalization)                  │
│         Formula: z = (x - μ) / σ                                │
│                                                                 │
│  4. HYPERPARAMETER TUNING                                       │
│     ├── Test K values from 1 to 39                              │
│     ├── Calculate mean error for each K                         │
│     └── Plot error rate vs K value                              │
│                                                                 │
│  5. MODEL TRAINING                                              │
│     └── Train KNN with K=8, Euclidean distance                  │
│                                                                 │
│  6. EVALUATION                                                  │
│     ├── Predict on scaled test set                              │
│     └── Calculate accuracy, precision, recall, F1-score         │
│                                                                 │
│  7. PREDICTION                                                  │
│     └── Accept new employee data → Predict salary class         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 3 — Why Feature Scaling?

KNN relies on **distance calculations** between data points. Features with larger ranges (e.g., `capital.gain`: 0–99,999) would dominate the distance computation over features with smaller ranges (e.g., `education.num`: 1–16). **StandardScaler** normalizes all features to have **mean = 0** and **standard deviation = 1**, ensuring each feature contributes equally.

$$z = \frac{x - \mu}{\sigma}$$

---

## 📈 Results & Performance

### Overall Accuracy

| Metric       | Value        |
|--------------|--------------|
| **Accuracy** | **81.51%**   |

### Classification Report

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| ≤50K    | 0.83      | 0.95   | 0.89     | 4,976   |
| >50K    | 0.70      | 0.38   | 0.49     | 1,537   |
| **Weighted Avg** | **0.80** | **0.82** | **0.79** | **6,513** |

### Confusion Matrix

|                   | Predicted ≤50K | Predicted >50K |
|-------------------|----------------|----------------|
| **Actual ≤50K**   | 4,731 (TN)     | 245 (FP)       |
| **Actual >50K**   | 959 (FN)       | 578 (TP)       |

### Key Observations

- **High recall for ≤50K class (95%)** — The model is very good at identifying individuals earning ≤$50K
- **Lower recall for >50K class (38%)** — The model misses a significant number of high earners, likely due to the class imbalance (76% vs 24%)
- **Precision for >50K (70%)** — When the model predicts >50K, it is correct 70% of the time

### Train-Test Split

| Set        | Records | Percentage |
|------------|---------|------------|
| Training   | 26,048  | 80%        |
| Testing    | 6,513   | 20%        |

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/Salary-Estimation-KNN---ML.git
cd Salary-Estimation-KNN---ML
```

### Step 2 — Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Step 3 — Run the Script

```bash
python Salary_Estimation.py
```

### Step 4 — Make Predictions

When prompted, enter the following values for a new employee:

```
Enter age: 35
Enter education level (num): 13
Enter capital gain: 5000
Enter hours per week: 40
```

The model will output whether the employee is likely to earn above or below $50K.

### Education Level Reference

| Value | Education Level     |
|-------|---------------------|
| 1     | Preschool           |
| 2     | 1st–4th Grade       |
| 3     | 5th–6th Grade       |
| 4     | 7th–8th Grade       |
| 5     | 9th Grade           |
| 6     | 10th Grade          |
| 7     | 11th Grade          |
| 8     | 12th Grade          |
| 9     | High School Grad    |
| 10    | Some College        |
| 11    | Associate (Voc)     |
| 12    | Associate (Acdm)    |
| 13    | Bachelors           |
| 14    | Masters             |
| 15    | Professional School |
| 16    | Doctorate           |

---

## 🛠️ Technologies Used

| Technology        | Purpose                              | Version |
|-------------------|--------------------------------------|---------|
| **Python**        | Programming language                 | 3.7+    |
| **NumPy**         | Numerical computations               | Latest  |
| **Pandas**        | Data manipulation & analysis         | Latest  |
| **Scikit-learn**  | ML model, preprocessing, evaluation  | Latest  |
| **Matplotlib**    | Data visualization (error rate plot) | Latest  |

---

## 📁 Project Structure

```
Salary-Estimation-KNN---ML/
│
├── Salary_Estimation.py    # Main Python script (training, evaluation, prediction)
├── salary.csv              # Dataset — Adult Census Income (32,561 records)
└── README.md               # Project documentation (this file)
```

---

## 🔮 Future Improvements

- **Handle class imbalance** — Apply SMOTE (Synthetic Minority Oversampling Technique) or class weighting to improve recall for the >50K class
- **Add more features** — Include occupation, marital status, race, and gender from the full Adult dataset for better predictions
- **Try other algorithms** — Compare KNN with Random Forest, SVM, Logistic Regression, and Gradient Boosting
- **Cross-validation** — Use K-Fold cross-validation instead of a single train-test split for more robust evaluation
- **Build a web interface** — Deploy the model using Flask or Streamlit for an interactive UI
- **Feature engineering** — Create new features like income-to-hours ratio or age groups for improved performance

---

## 📜 License

This project is open-source and available for educational purposes.

---

## 👤 Author

Built as a Machine Learning practice project for salary prediction using the KNN classification algorithm.