# Logistic Regression from Scratch

I built this project while learning Logistic Regression to understand how the algorithm works internally instead of relying completely on a pre-built model.

## Dataset

UCI Automobile Dataset

Features used:

* Engine size
* Horsepower
* Highway MPG

I converted the car price into a binary target using the median price:

* `0` → Below median price
* `1` → At or above median price

## What I Built

I implemented Logistic Regression from scratch using NumPy:

* Sigmoid function
* Logistic cost function
* Gradient calculation
* Gradient Descent
* Probability predictions
* 0.5 classification threshold
* Train/test evaluation

## Results

Test Accuracy: **87.8%**

Precision: **85.0%**

Recall: **89.5%**

F1 Score: **~87%**

Confusion Matrix:

**
                 Predicted
              0          1
Actual  0    TN=19      FP=3
        1    FN=2       TP=17
***

The cost decreased from approximately **0.690 to 0.514** during training.

## What I Learned

This project helped me understand the complete Logistic Regression process:


***Features → X @ w + b → Sigmoid → Probability → 0.5 Threshold → Prediction


I also got more comfortable with NumPy, vectorised calculations, Gradient Descent and evaluating classification models.

## Tools

Python, NumPy, pandas, scikit-learn, Matplotlib
