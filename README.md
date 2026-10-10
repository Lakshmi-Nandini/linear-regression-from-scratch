# Linear Regression: From Scratch vs Scikit-Learn

## 📌 Project Overview

This project demonstrates Linear Regression through two implementations:

1. **From Scratch:** Implementing Linear Regression using NumPy and Gradient Descent.
2. **Scikit-Learn:** Training a Linear Regression model using Scikit-Learn's built-in `LinearRegression` estimator.

The main objective is to understand the mathematical foundations of Linear Regression, how model parameters are learned, and how a manually implemented algorithm compares with a standard machine learning library.

## 🎯 Objectives

- Understand the mathematics behind Linear Regression.
- Implement prediction, error calculation, and Mean Squared Error (MSE).
- Calculate gradients and update weights and bias using Gradient Descent.
- Understand feature standardization and its importance in optimization.
- Evaluate model performance on training and test data.
- Compare a from-scratch implementation with Scikit-Learn.

## 📂 Project Structure

```text
linear-regression-from-scratch/
│
├── linear_regression_from_scratch.ipynb
├── linear_regression_sklearn.ipynb
├── README.md
└── requirements.txt
```

## 📊 Dataset

A small illustrative house-price dataset is used to understand the algorithm.

**Input features:**
- Area in square feet
- Number of bedrooms

**Target variable:**
- House price in illustrative units

The dataset contains five training samples and three test samples. It is intentionally small to make the mathematical calculations and learning process easier to understand.

**Note:** This dataset is intended for educational purposes and is not suitable for real-world house-price prediction.

## 🧮 Mathematical Foundation

### 1. Linear Regression Equation

The model predicts the target value using a linear combination of input features:

ŷ = w₁x₁ + w₂x₂ + b

Where:

- ŷ = predicted house price
- x₁ = area of the house
- x₂ = number of bedrooms
- w₁ and w₂ = learned weights
- b = bias or intercept

### 2. Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted values.

MSE = (1/n) × Σ(yᵢ − ŷᵢ)²

A smaller MSE indicates lower prediction error on the evaluated dataset.

### 3. Gradient Descent

Gradient Descent iteratively updates the model parameters to minimize the loss function.

Weight update:

w := w − α × ∂MSE/∂w

Bias update:

b := b − α × ∂MSE/∂b

Where α represents the learning rate, which controls the size of each parameter update.

## ⚙️ Implementation 1: From Scratch

The first notebook implements the following steps using Python and NumPy:

1. Define the training and test datasets.
2. Calculate feature means and standard deviations using training data.
3. Standardize the training and test features using the training statistics.
4. Initialize weights and bias.
5. Calculate predictions using the Linear Regression equation.
6. Calculate prediction errors and MSE.
7. Compute gradients for the weights and bias.
8. Update parameters using Gradient Descent.
9. Repeat the training process for multiple epochs.
10. Evaluate predictions using the test dataset.
11. Visualize the training loss over epochs.

**Notebook:** `linear_regression_from_scratch.ipynb`

## 🤖 Implementation 2: Scikit-Learn

The second notebook uses Scikit-Learn's `LinearRegression` estimator.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The learned parameters can be accessed using:

```python
print(model.coef_)
print(model.intercept_)
```

Scikit-Learn's implementation solves an ordinary least-squares problem using a numerical least-squares solver rather than the explicit Gradient Descent loop implemented in the first notebook.

**Notebook:** `linear_regression_sklearn.ipynb`

## 📈 Model Evaluation

Both implementations are evaluated using Mean Squared Error.

The project evaluates:
- Training error
- Test error
- Predicted values compared with actual values
- Training loss across epochs for the from-scratch model

The numerical results are available in the respective notebooks.

Since the dataset is very small and deliberately constructed, the results should be interpreted as a demonstration of the algorithm rather than evidence of real-world generalization.

## 🔍 Key Learnings

Through this project, I learned:

- How Linear Regression represents relationships between features and a target variable.
- How weights and bias determine predictions.
- How MSE measures prediction error.
- How gradients guide parameter updates during training.
- Why feature standardization can improve Gradient Descent optimization.
- Why training-set statistics must be reused when transforming test data.
- How a manually implemented model compares with Scikit-Learn's implementation.
- Why evaluating a model on unseen data is important.

## 🛠️ Technologies Used

- Python
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-from-scratch.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Navigate to the project directory

```bash
cd linear-regression-from-scratch
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Open either notebook and run the cells from top to bottom.

Alternatively, upload the notebooks to Google Colab and execute them there.

## 🔮 Future Improvements

- Experiment with a larger, publicly available house-price dataset.
- Compare model performance against a baseline.
- Visualize actual versus predicted values.
- Analyze residuals and different evaluation metrics.
- Experiment with different learning rates and numbers of epochs.
- Explore additional Linear Regression implementations and optimization methods.

## 👩‍💻 Author

I am building my Machine Learning portfolio by implementing algorithms from their mathematical foundations and comparing them with established machine learning libraries.

This project is part of my ongoing journey to develop practical Machine Learning and AI engineering skills.
