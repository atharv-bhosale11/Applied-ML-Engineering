# 🏏🎾 Sports Ball Classification using Decision Tree

A Machine Learning classification project that predicts whether a sports ball is a **Tennis Ball** or a **Cricket Ball** based on its physical characteristics such as weight and surface texture.

This project demonstrates the implementation of a **Decision Tree Classifier** using Scikit-Learn for supervised machine learning classification tasks.

---

## 🚀 Project Objective

The objective of this project is to classify sports balls using:

- Weight of the ball
- Surface texture (Rough / Smooth)

The model learns patterns from training data and predicts the category of a new ball.

---

## 🛠️ Technologies Used

- Python
- Scikit-Learn
- Decision Tree Classifier

---

## 📊 Dataset Description

### Features

| Feature | Description |
|----------|-------------|
| Weight | Weight of the ball |
| Surface Texture | Rough or Smooth |

### Encoding

| Value | Meaning |
|---------|---------|
| Rough | 1 |
| Smooth | 0 |

### Target Variable

| Value | Ball Type |
|---------|-----------|
| 1 | Tennis Ball |
| 2 | Cricket Ball |

---

## 🌳 Machine Learning Algorithm

### Decision Tree Classifier

The project uses a Decision Tree Classifier to learn patterns from the training dataset and classify unseen samples.

---

## 📂 Project Structure

```text
01-Sports-Ball-Classification
│
├── ball_classification.py
├── screenshots/
│   └── prediction_output.png
│
└── README.md
```

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install scikit-learn
```

### Execute Program

```bash
python ball_classification.py
```

---

## 📈 Sample Prediction

Input:

```python
[35,1]
```

Where:

- Weight = 35
- Surface = Rough

Output:

```text
Object looks like Tennis Ball
```

---

## 🔍 Workflow

1. Prepare training dataset.
2. Encode categorical values.
3. Train Decision Tree model.
4. Predict ball category.
5. Display prediction result.

---

## 📸 Output Screenshots

Store execution screenshots inside:

```text
screenshots/
```

Example:

```text
screenshots/
└── prediction_output.png
```

---

## 🎯 Learning Outcomes

- Supervised Machine Learning
- Classification Problems
- Feature Representation
- Decision Tree Algorithm
- Model Training and Prediction
- Scikit-Learn Basics

---

## 🚀 Future Enhancements

- Add larger dataset
- Calculate model accuracy
- Add confusion matrix
- Add visualization of decision tree
- Support additional sports ball categories

---

## 👨‍💻 Author

### Atharv Tushar Bhosale

Machine Learning and Software Engineering Enthusiast

GitHub:
https://github.com/atharv-bhosale11

---

⭐ If you found this project useful, consider giving it a star.
