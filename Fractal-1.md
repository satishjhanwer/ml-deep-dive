# Section 1: Supervised Learning

## 1. Introduction to AI and ML

### Artificial Intelligence (AI)

- **Definition**: AI is the simulation of human intelligence by machines to perform tasks that would typically require human intelligence, such as decision-making, visual perception, and natural language understanding.
- **Types of AI**:
  - **Narrow AI**: Specialized systems focused on a single task (e.g., image recognition).
  - **General AI**: Hypothetical system capable of performing any human-level intellectual task.

### Machine Learning (ML)

- **Definition**: A subset of AI that enables machines to improve their performance on a specific task through experience/data, without being explicitly programmed.
- **Core Principle**: Algorithms learn patterns from data, adjusting based on feedback to optimize their predictive or decision-making accuracy.

---

## 2. Machine Learning Paradigms

Machine Learning is broadly categorized into three main paradigms based on how models learn from data:

### 1. Supervised Learning

- **Definition**: A learning approach where the model is trained using labeled data. Each data instance has input features and a known output label.
- **Goal**: To learn a mapping from inputs to the output labels and generalize to unseen data.
- **Examples**:
  - **Classification**: Identifying if an email is spam or not (output is categorical).
  - **Regression**: Predicting house prices based on features like size and location (output is continuous).

### 2. Unsupervised Learning

- **Definition**: Here, the model is trained on unlabeled data. The algorithm attempts to identify patterns or structure within the data.
- **Goal**: To discover the underlying structure, often used in clustering or association tasks.
- **Examples**:
  - **Clustering**: Grouping customers by purchasing behavior.
  - **Dimensionality Reduction**: Simplifying data by reducing the number of features while retaining essential information.

### 3. Reinforcement Learning (RL)

- **Definition**: A learning method where an agent interacts with an environment and learns to perform actions to maximize cumulative rewards over time.
- **Goal**: To learn optimal actions or policies through a feedback loop of rewards and penalties.
- **Examples**:
  - **Game Playing**: Learning strategies in games like Chess or Go.
  - **Robotics**: Training robots to perform tasks like walking or picking up objects.

## 3. Bayesian Classification

### Overview

- **Definition**: Bayesian classification uses Bayes’ Theorem to predict the probability that a data instance belongs to a particular class based on prior knowledge.
- **Bayes’ Theorem**:
  $$
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  $$
  where:
  - \( P(A|B) \) is the posterior probability of class \( A \) given evidence \( B \).
  - \( P(B|A) \) is the likelihood of evidence \( B \) given class \( A \).
  - \( P(A) \) is the prior probability of class \( A \).
  - \( P(B) \) is the probability of evidence \( B \).

### Naive Bayes Classifier

- **Assumption**: Assumes that features are independent given the class label.
- **Calculation**:
  - The classifier predicts class \( C_k \) for which \( P(C_k|X) \) is highest, where \( X = \{x_1, x_2, \dots, x_n\} \).
  - Using Bayes' theorem:
    $$
    P(C_k|X) = \frac{P(X|C_k) \cdot P(C_k)}{P(X)}
    $$
  - **Naive Bayes Formula** (assuming feature independence):
    $$
    P(C*k|X) = P(C_k) \prod*{i=1}^n P(x_i | C_k)
    $$

### Advantages and Disadvantages

- **Advantages**:
  - Simple and efficient, especially for large datasets.
  - Performs well in text classification tasks (e.g., spam detection).
- **Disadvantages**:
  - The independence assumption may not hold true in many real-world scenarios.
  - Struggles with data where features are highly correlated.

---

## 4. Decision Tree Learning

### Overview

- **Definition**: A decision tree is a tree-like model of decisions used for classification and regression.
- **Structure**:
  - **Nodes**: Represent features/attributes.
  - **Edges**: Represent decisions based on attribute values.
  - **Leaves**: Represent class labels or outcomes.

### Decision Tree Algorithm

1. **Selection of Splits**:
   - At each node, the best attribute is chosen to split the data based on a measure of "impurity" (e.g., Information Gain, Gini Index).
2. **Impurity Measures**:

   - **Information Gain** (Entropy-based):
     $$
     IG = Entropy(parent) - \sum \left(\frac{n_k}{n}\right) \cdot Entropy(child_k)
     $$
   - **Gini Index**:
     $$
     Gini(D) = 1 - \sum\_{k=1}^K p_k^2
     $$
     where \( p_k \) is the proportion of instances in class \( k \).

3. **Tree Construction**:
   - Recursively split nodes based on impurity until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf).

### Advantages and Disadvantages

- **Advantages**:
  - Easy to interpret and visualize.
  - Can handle both numerical and categorical data.
- **Disadvantages**:
  - Prone to overfitting, especially with deep trees.
  - Can be unstable; small changes in data may lead to a different tree structure.

## 5. Ensemble Methods

### Overview

- **Definition**: Ensemble methods combine predictions from multiple models to create a more accurate and robust prediction.
- **Goal**: Improve performance by reducing variance (Bagging), bias (Boosting), or leveraging diverse model strengths (Stacking).

---

### 5.1 Bagging (Bootstrap Aggregating)

- **Idea**: Reduces variance by training multiple models on different subsets of the training data, then averaging or voting on their predictions.
- **Process**:
  1. Generate multiple subsets of the training data through bootstrapping (random sampling with replacement).
  2. Train a model on each subset independently.
  3. Aggregate predictions (e.g., majority vote for classification, average for regression).
- **Example Algorithm**: Random Forest.

  - A Random Forest creates a collection of decision trees trained on bootstrapped samples, with additional randomness introduced in feature selection.

- **Advantages**:
  - Reduces overfitting by averaging predictions across models.
  - Works well with high-variance models like decision trees.
- **Disadvantages**:
  - Can be computationally expensive, especially with many models.

---

### 5.2 Boosting

- **Idea**: Sequentially trains models, each attempting to correct the errors of its predecessor, which reduces bias and can create a stronger model overall.
- **Process**:
  1. Start with a base model and calculate its errors.
  2. Adjust the weights of misclassified samples to focus the next model on harder cases.
  3. Train the next model on the updated dataset, iteratively repeating the process.
  4. Aggregate all models, typically by weighted voting or weighted average.
- **Example Algorithms**:

  - **AdaBoost**: Adjusts sample weights to emphasize misclassified points, combining weak learners.
  - **Gradient Boosting**: Models residual errors iteratively to minimize loss function.

- **Advantages**:

  - Often achieves high accuracy, especially on complex datasets.
  - Can convert weak learners into a strong predictive model.

- **Disadvantages**:
  - Sensitive to noisy data and outliers, as boosting focuses on hard-to-classify cases.
  - More prone to overfitting than bagging if not carefully tuned.

---

### 5.3 Stacking (Stacked Generalization)

- **Idea**: Combines different types (or levels) of models by training a "meta-model" on their predictions, allowing it to learn which models perform best under various conditions.
- **Process**:
  1. Split the training data into two parts.
  2. Train multiple base models (e.g., decision tree, SVM, neural network) on the first part.
  3. Use the predictions from these base models on the second part as inputs to train a "meta-model."
  4. The meta-model learns to optimize the combination of base model predictions.
- **Example**: Using logistic regression as a meta-model to combine predictions from a decision tree, SVM, and k-NN classifiers.

- **Advantages**:

  - Leverages the strengths of different models, often leading to higher performance.
  - Flexible as it can combine models with different underlying assumptions.

- **Disadvantages**:
  - Complex to implement and computationally intensive.
  - Requires careful tuning to avoid overfitting, especially for the meta-model.

## Examples of Ensemble Methods

### 1. Bagging Example: Random Forest

- **Scenario**: Predicting if a customer will churn based on attributes like age, contract type, monthly charges, etc.
- **Process**:
  1. **Create Bootstrapped Datasets**: From the original dataset of customer information, create multiple subsets by random sampling with replacement.
  2. **Train Decision Trees**: Train a decision tree on each subset. In a random forest, each tree is also given a random subset of features to reduce correlation among trees.
  3. **Aggregate Predictions**:
     - For classification: Each tree votes on whether the customer will churn; the majority vote determines the final prediction.
     - For regression: The predictions from each tree are averaged for a final continuous prediction (e.g., predicted monthly charges).
- **Benefit**: By averaging many trees, the random forest reduces variance, often resulting in a more stable prediction than a single tree.

---

### 2. Boosting Example: AdaBoost

- **Scenario**: Classifying emails as spam or not spam using features like word frequency, sender info, etc.
- **Process**:
  1. **Start with Equal Weights**: Assign equal weight to all training samples initially.
  2. **Train Weak Learners**: Train a weak learner (e.g., a shallow decision tree, also called a "stump").
  3. **Focus on Misclassified Samples**: Increase the weights of misclassified samples so the next learner focuses on correcting these errors.
  4. **Combine Predictions**: After several weak learners are trained, combine their predictions, weighting each based on its accuracy (more accurate learners have more influence).
- **Final Prediction**: The model aggregates predictions across all weak learners to classify an email as spam or not spam.
- **Benefit**: Boosting can build a powerful classifier from multiple weak learners, making it highly accurate.

---

### 3. Stacking Example

- **Scenario**: Predicting loan defaults using three models—logistic regression, a decision tree, and a k-nearest neighbors (k-NN) classifier.
- **Process**:
  1. **Split Dataset**: Divide the training dataset into two parts.
  2. **Train Base Models**:
     - Train logistic regression, decision tree, and k-NN classifiers on the first part of the dataset.
     - Record their predictions on the second part.
  3. **Train Meta-Model**: Using the predictions from the base models on the second part as features, train a meta-model (e.g., a logistic regression model).
  4. **Final Prediction**: For new data, each base model provides a prediction, which the meta-model then uses to make the final prediction.
- **Benefit**: Stacking leverages the strengths of each model type, often resulting in better overall performance.
