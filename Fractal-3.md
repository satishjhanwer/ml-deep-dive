# Section 3: Unsupervised Learning and Feature Selection

## 1. Unsupervised Learning

### Overview

- **Definition**: Unsupervised learning involves training models on data without labeled outputs, enabling the discovery of hidden patterns or intrinsic structures in the data.
- **Common Goals**:
  - **Clustering**: Grouping similar data points together.
  - **Dimensionality Reduction**: Reducing the number of features while preserving essential information.

### Key Techniques

#### 1.1 Clustering

- **Definition**: The task of partitioning a dataset into groups (clusters) such that data points in the same group are more similar to each other than to those in other groups.

##### Examples of Clustering Algorithms

1. **k-Means Clustering**

   - **Process**:
     1. Choose the number of clusters \( k \).
     2. Initialize \( k \) centroids randomly.
     3. Assign each data point to the nearest centroid.
     4. Update centroids by computing the mean of all points in each cluster.
     5. Repeat steps 3 and 4 until convergence (no changes in assignments).
   - **Usage**: Customer segmentation in marketing based on purchasing behavior.

2. **k-Medoids**

   - **Definition**: Similar to k-means but uses actual data points (medoids) as cluster centers.
   - **Usage**: More robust to noise and outliers than k-means.

3. **Expectation-Maximization (EM) Algorithm**

   - **Definition**: A statistical approach for finding the maximum likelihood estimates of parameters in probabilistic models, commonly used for clustering with Gaussian Mixture Models (GMM).
   - **Process**:
     1. **E-step**: Estimate the probabilities of the data points belonging to each cluster.
     2. **M-step**: Update the parameters (means and covariances) based on the estimated probabilities.
   - **Usage**: Identifying subpopulations in a dataset with underlying Gaussian distributions.

4. **Agglomerative Clustering**
   - **Definition**: A hierarchical clustering method that starts with each point as a single cluster and merges them iteratively based on distance criteria.
   - **Process**:
     1. Compute the distance matrix for all data points.
     2. Merge the two closest clusters.
     3. Update the distance matrix and repeat until a single cluster remains or a stopping criterion is met.
   - **Usage**: Document clustering based on content similarity.

---

## 2. Feature Selection and Dimensionality Reduction

### Overview

- **Definition**: Techniques used to select a subset of relevant features (variables) for model training, or to reduce the dimensionality of the dataset while preserving important information.
- **Goals**:
  - Improve model performance.
  - Reduce overfitting.
  - Decrease training time.

### Key Techniques

#### 2.1 Feature Selection

- **Definition**: The process of selecting a subset of relevant features for model training from a larger set of available features.

##### Methods of Feature Selection

1. **Filter Methods**

   - **Definition**: Evaluate the importance of features using statistical measures and select the top-ranked features.
   - **Example**: Using correlation coefficients to identify features strongly correlated with the target variable.
   - **Usage**: Fast and scalable; suitable for high-dimensional datasets.

2. **Wrapper Methods**

   - **Definition**: Use a predictive model to evaluate feature subsets based on their performance.
   - **Example**: Recursive Feature Elimination (RFE) removes the least important features iteratively until the desired number is reached.
   - **Usage**: More computationally intensive; often leads to better performance but may risk overfitting.

3. **Embedded Methods**
   - **Definition**: Perform feature selection as part of the model training process.
   - **Example**: Lasso regression adds a penalty to the loss function, shrinking coefficients of less important features to zero.
   - **Usage**: Provides a balance between filter and wrapper methods.

---

#### 2.2 Dimensionality Reduction

- **Definition**: Techniques to reduce the number of features in a dataset while retaining as much information as possible.

##### Examples of Dimensionality Reduction Techniques

1. **Principal Component Analysis (PCA)**

   - **Process**:
     1. Standardize the data to have zero mean and unit variance.
     2. Compute the covariance matrix.
     3. Calculate eigenvalues and eigenvectors of the covariance matrix.
     4. Select the top \( k \) eigenvectors (principal components) that capture the most variance.
     5. Transform the data into the new feature space defined by the selected components.
   - **Usage**: Data visualization, noise reduction, and improving model performance by removing redundant features.

2. **Linear Discriminant Analysis (LDA)**

   - **Definition**: A supervised technique that finds the linear combinations of features that best separate classes.
   - **Usage**: Effective in reducing dimensionality while preserving class separability, often used in classification problems.

3. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
   - **Definition**: A non-linear dimensionality reduction technique primarily used for visualizing high-dimensional datasets.
   - **Process**: Maintains local structures while reducing dimensionality, making it suitable for clustering and classification tasks.
   - **Usage**: Visualizing high-dimensional data like images in 2D or 3D space.

## Examples for Section 3: Unsupervised Learning and Feature Selection

### 1. Clustering

#### 1.1 k-Means Clustering Example

- **Scenario**: Customer Segmentation
- **Process**:
  1. **Data**: Features such as age, income, and spending score of customers.
  2. **Choose k**: Decide to segment customers into 3 groups.
  3. **Initialization**: Randomly select 3 initial centroids.
  4. **Assignment**: Assign each customer to the nearest centroid based on Euclidean distance.
  5. **Update**: Calculate new centroids as the mean of assigned customers and repeat until centroids no longer change.
- **Outcome**: Identified segments (e.g., low-income spenders, high-income spenders, budget-conscious) can help tailor marketing strategies.

---

#### 1.2 Agglomerative Clustering Example

- **Scenario**: Document Clustering
- **Process**:
  1. **Data**: A collection of news articles.
  2. **Distance Matrix**: Compute pairwise similarities between articles using cosine similarity.
  3. **Merging**: Start with each article as a cluster and iteratively merge the closest pairs until a specified number of clusters is reached.
- **Outcome**: Grouped articles into clusters based on topic, facilitating easier retrieval and organization of related content.

---

### 2. Feature Selection

#### 2.1 Filter Method Example: Correlation Coefficient

- **Scenario**: Predicting House Prices
- **Process**:
  1. **Data**: A dataset containing features such as square footage, number of bedrooms, and distance to city center.
  2. **Calculate Correlation**: Use Pearson correlation to find the correlation between each feature and the target variable (house price).
  3. **Select Features**: Identify and keep features with a correlation coefficient above a certain threshold (e.g., 0.5).
- **Outcome**: Reduced feature set might include square footage and number of bedrooms, improving model interpretability and performance.

---

#### 2.2 Wrapper Method Example: Recursive Feature Elimination (RFE)

- **Scenario**: Classifying Emails as Spam
- **Process**:
  1. **Initial Model**: Train a classification model (e.g., logistic regression) using all features.
  2. **Evaluate Importance**: Rank features based on their contribution to the model's performance (e.g., using feature importance scores).
  3. **Elimination**: Remove the least important feature and retrain the model, repeating until the desired number of features is reached.
- **Outcome**: A smaller set of features leading to improved model accuracy and reduced overfitting.

---

### 3. Dimensionality Reduction

#### 3.1 Principal Component Analysis (PCA) Example

- **Scenario**: Image Compression
- **Process**:
  1. **Data**: A dataset of high-resolution images.
  2. **Standardization**: Normalize pixel values to have zero mean and unit variance.
  3. **Covariance Matrix**: Compute the covariance matrix of the data.
  4. **Eigenvalues and Eigenvectors**: Calculate eigenvalues and corresponding eigenvectors.
  5. **Select Components**: Choose the top \( k \) eigenvectors (e.g., the first 10) that explain the most variance.
  6. **Transform Data**: Project original images onto the new space defined by the selected eigenvectors.
- **Outcome**: Reduced dimensionality leads to smaller file sizes while retaining most of the image's original information, useful for storage and faster processing.

---

#### 3.2 t-Distributed Stochastic Neighbor Embedding (t-SNE) Example

- **Scenario**: Visualizing High-Dimensional Data
- **Process**:
  1. **Data**: A high-dimensional dataset such as the MNIST dataset of handwritten digits (28x28 pixel images).
  2. **t-SNE Application**: Apply t-SNE to reduce the 784 dimensions (28x28 pixels) down to 2 or 3 dimensions for visualization.
  3. **Visualization**: Create a scatter plot of the reduced data points.
- **Outcome**: Clusters of similar digits appear in the plot, helping to understand the data distribution and potential separability of classes.
