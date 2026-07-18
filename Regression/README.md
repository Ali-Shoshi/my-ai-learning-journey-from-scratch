# Regression

1. ## Linear Decision Boundary
    - **Project Goal:** The algorithm learns whether the sum of two numbers is greater than or less than 100 using supervised learning.
    - **Classification Rules:**  
        >1 -> (x1 + x2 > 100)  
        0 -> (x1 + x2 < 100)
    - **Implementation Details:** It implements a single-layer Perceptron from scratch to find the optimal weights and bias for a linear decision boundary. The model is trained on a synthetic dataset of 10,000 generated coordinates $(x_1, x_2)$, where each point is labeled based on whether its sum exceeds a threshold of 100 ($x_1 + x_2 > 100$). By utilizing a basic threshold activation function and adjusting the weights ($w_1, w_2, w_3$) incrementally through supervised learning steps governed by a learning rate ($\alpha = 0.0005$), the algorithm iteratively minimizes classification errors over 40 epochs. Ultimately, the code successfully learns to approximate the mathematical boundary dividing the two data classes.
2. ## Nonelinear Regression Carprice
    - **Project Goal:** The algorithm learns to predict the resale price of a used car based on three input features—purchase price, age, and mileage—using supervised non-linear regression.
    - **Mathematical Framework:** The model utilizes an exponential decay structure to mirror real-world automotive depreciation, ensuring that estimated asset valuations scale realistically and never drop below zero. It is expressed as:
      > $$\Large predictedPrice = w_0 \cdot \left(\frac{price}{300000}\right) \cdot e^{-w_1 \cdot \left(\frac{year}{30}\right)} \cdot e^{-w_2 \cdot \left(\frac{mileage}{300000}\right)} \cdot 300000$$
    - **Implementation Details:** It implements a multi-variable non-linear regression model trained from scratch using gradient descent to minimize an L2 Squared Error loss function. The training pipeline automatically generates a synthetic dataset of 2,000 car records with known target parameters ($w_0 = 1.0, w_1 = 3.2, w_2 = 0.08$). To ensure mathematical stability and prevent gradient explosion, all features are normalized via feature scaling during execution. By calculating exact partial derivatives through the calculus Chain Rule and updating the weights ($w_0, w_1, w_2$) incrementally using a learning rate ($\alpha = 0.05$) over 100 epochs, the model successfully converges onto the true underlying parameters and minimizes Mean Absolute Error (MAE).
3. ## Linear Regression With Noise L2 Regularization
    - **Project Goal:** The algorithm learns to approximate a continuous linear baseline relationship from a noisy synthetic dataset using supervised multiple linear regression.
    - **Mathematical Framework:** The model uses a classic linear combination equation to predict targets, adding an explicit $L_2$ regularization penalty (Weight Decay) to actively suppress data noise during optimization. It is expressed as:
      > $$\Large y = w_0 \cdot x + w_1$$
    - **Implementation Details:** It implements a multi-variable linear regression model trained from scratch using Batch Gradient Descent to minimize a Mean Squared Error (MSE) loss metric. The framework automatically streams a dataset of 10,000 synthetic coordinate records based on a hidden ground truth equation ($w_0 = 52.0, w_1 = 13.0$) and injects uniform additive random noise within the closed bounds of $[-5.0, 5.0]$. By calculating precise partial derivatives over the full data batch per iteration, managing floating-point tracking variables to eliminate integer division truncations, and applying a steady decay pressure via a regularization modifier ($\lambda = 0.001$) across 500 training epochs, the algorithm filters past chaotic fluctuations to successfully mirror the original target weights and hit a near-optimal Mean Absolute Error (MAE) limit of $\sim 2.74$.
4. ## k-Nearest Neighbors (k-NN) Three-Circle Classifier
    - **Project Goal:** The algorithm learns to classify multi-class spatial data points across geometric distributions by analyzing local structural density and assigning category labels based on spatial proximity.
    - **Mathematical Framework:** The model is non-parametric and relies on distance-based localized voting rather than explicit weights. To evaluate proximity between an unlabelled query point $q$ and a reference dataset point $p$ in a two-dimensional Euclidean vector space, the engine calculates the standard geometric L2 distance metric. This is mathematically defined as:
        > $$\Large \text{distance}(q, p) = \sqrt{(q.x - p.x)^2 + (q.y - p.y)^2}$$
        >
        > $$\Large P(y = c \mid q) = \frac{1}{k} \sum_{i \in N_k(q)} I(y_i = c)$$
    - **Implementation Details:** This system implements a $k$-Nearest Neighbors ($k$-NN) instance-based classifier built entirely from scratch in C++. The training pipeline automatically synthesizes a dataset containing 120 geometric coordinate clusters distributed symmetrically across three overlapping circular spaces centered at distinct coordinates ($(0, 0)$, $(3, 0)$, and $(1.5, 2.598)$). During execution, the framework calculates the Euclidean distance from a target query to every spatial coordinate vector in the memory matrix, sorts the results using a quick-sort variant, and isolates the top $k$ closest neighbors ($k=5$). The output label is determined via a majority plurality vote from the selected neighbors, enabling highly accurate non-linear classification boundaries without requiring a formal iterative training phase.
5. ## Polynomial Logistic Regression Day/Night Classifier
    - **Project Goal:** The algorithm learns to classify a given hour of the day as either daytime (light) or nighttime (dark) using supervised polynomial logistic regression, successfully capturing multiple temporal thresholds simultaneously.
    - **Mathematical Framework:** Because a single-neuron model cannot inherently separate non-linear data structures (where an island of light is trapped between two periods of darkness), the algorithm expands the input space using a polynomial expansion. By calculating a quadratic feature component ($\text{hour}^2$), the model wraps a parabolic decision boundary around the daylight parameters. The raw hypothesis is pushed through a Sigmoid activation function to map real values to a stable probability distribution between $[0, 1]$. It is expressed as:
        > $$\Large Z = w_0 + w_1 \cdot \left(\frac{\text{hour}}{24}\right) + w_2 \cdot \left(\frac{\text{hour}}{24}\right)^2$$
        >
        > $$\Large Z = w_0 + w_1 \cdot X_1 + w_2 \cdot X_2^2$$
        >
        > $$\Large \text{predicted (Z)} = \frac{1}{1 + e^{-z}}$$
        >
        > $$Loss = - \big[ y \ln(predicted) + (1 - y) \ln(1 - predicted) \big]$$
        >
        > $$\frac{\partial \text{Loss}}{\partial w_j} = \frac{\partial \text{Loss}}{\partial \text{predicted}} \times \frac{\partial \text{predicted}}{\partial z} \times \frac{\partial z}{\partial w_j}$$
        >
        > $$\frac{\partial \text{Loss}}{\partial w_j} = \frac{\text{predicted} - y}{\cancel{\text{predicted}(1 - \text{predicted})}} \times \cancel{\text{predicted}(1 - \text{predicted})} \times \frac{\partial z}{\partial w_j}$$
        >
        > $$\frac{\partial \text{Loss}}{\partial w_j} = (\text{predicted} - y) \times \frac{\partial z}{\partial w_j}$$
        >
        > ---
        >
        > $$w_j \leftarrow w_j - \alpha \cdot \frac{\partial \text{Loss}}{\partial w_j} $$
        >
        > $$w_0 \leftarrow w_0 - \alpha \cdot (predicted - y)$$
        >
        > $$w_1 \leftarrow w_1 - \alpha \cdot (predicted - y) \cdot x$$
        >
        > $$w_2 \leftarrow w_2 - \alpha \cdot (predicted - y) \cdot x^2$$
    - **Implementation Details:** The engine implements a parametric polynomial logistic regression network built entirely from scratch in. The training pipeline dynamically generates a synthetic matrix of 10,000 distinct timestamps, automatically establishing ground-truth boundaries between sunrise (6.00) and sunset (18.00). To avoid gradient saturation across the Sigmoid curves and ensure smooth numeric step adjustments, time values are scaled directly down to a normalized $[0, 1]$ decimal range. Utilizing Stochastic Online Learning paired with Gradient Descent, the network iteratively tunes the feature parameters ($w_0, w_1, w_2$) using a learning rate ($\alpha = 0.01$) over 100 epochs, allowing the model to conform perfectly to the parabolic thresholds and maximize evaluation classification accuracy.