# Neural Network

1. ## Multi-Layer Perceptron XOR Gate Classifier
    - **Project Goal:** The algorithm learns to resolve the classic non-linearly separable XOR logical function, training a multi-layer neural network from scratch to map binary input coordinate pairs into their correct single-bit parity outputs.
    - **Mathematical Framework:** Because a single-layer perceptron cannot construct a non-linear decision boundary to isolate the staggered true and false coordinates of an XOR truth table, the system utilizes a 2-2-1 feedforward architecture. Signals cascade through a hidden layer before reaching the output, with every neuron's dot product compressed by a non-linear Logistic Sigmoid function. During backpropagation, optimization gradients are computed via the chain rule to minimize the sum of squared errors ($E$). The error terms ($\delta$) are calculated at the output and distributed backward through the weight matrices using the first derivative of the activation function, expressed as:
        > $$\Large \text{Loss Function} = E = \frac{1}{2}(Y_s - \hat{Y}_s)^2$$
        >
        > $$\Large h_j = \sigma(z_{\text{hidden},j}) = \sigma\left(\sum_{i} (x_i \cdot w_{ij}) + b_j\right) = \frac{1}{1 + e^{-z_{\text{hidden},j}}}$$
        >
        > $`\hat{Y}_s = \sigma(z_{\text{output}}) = \sigma \left( \sum_{j} (h_j \cdot w_j) + b_{\text{output}} \right) = \frac{1}{1 + e^{-z_{\text{output}}}}`$
        >
        > $$\Large \frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z))$$
        >
        > ---
        >
        > $`\delta_{\text{output}} = -\frac{\partial E}{\partial z_{\text{output}}} = -\frac{\partial E}{\partial \hat{Y}_s} \cdot \frac{\partial \hat{Y}_s}{\partial z_{\text{output}}}`$
        > $$\Large = - \frac{\partial}{\partial \hat{Y}_s} \left[ \frac{1}{2}(Y_s - \hat{Y}_s)^2 \right] \cdot \hat{Y}_s(1 - \hat{Y}_s)$$
        >
        > $$\Large = -(-(Y_s - \hat{Y}_s)) \cdot \hat{Y}_s(1 - \hat{Y}_s) = (Y_s - \hat{Y}_s) \cdot \hat{Y}_s(1 - \hat{Y}_s)$$
        >
        > ---
        >
        > $`\delta_{\text{hidden},j} = -\frac{\partial E}{\partial z_{\text{hidden},j}} = -\frac{\partial E}{\partial \hat{Y}_s} \cdot \frac{\partial \hat{Y}_s}{\partial z_{\text{output}}} \cdot \frac{\partial z_{\text{output}}}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_{\text{hidden},j}}`$
        >
        > $$\Large = (Y_s - \hat{Y}_s) \cdot \hat{Y}_s(1 - \hat{Y}_s) \cdot w_j \cdot h_j(1 - h_j)$$
        >
        > $$\Large = \delta_{\text{output}} \cdot w_j \cdot h_j(1 - h_j)$$
        >
        > ---
        >
        > $$\Large \delta_{\text{output}} = (Y_s - \hat{Y}_s) \cdot \hat{Y}_s(1 - \hat{Y}_s)$$
        >
        > $$\Large \delta_{\text{hidden},j} = \delta_{\text{output}} \cdot w_j \cdot h_j(1 - h_j)$$
        >
        > ---
        >
        > $$\Large \Delta w_j = \alpha \cdot \delta_{\text{output}} \cdot h_j \qquad \text{and} \qquad \Delta b_{\text{output}} = \alpha \cdot \delta_{\text{output}}$$
        >
        > $$\Large \Delta w_{ij} = \alpha \cdot \delta_{\text{hidden},j} \cdot x_i \qquad \text{and} \qquad \Delta b_j = \alpha \cdot \delta_{\text{hidden},j}$$
        >
        > ---
        >
        > * $E$ is the total squared error for a single training sample.
        > * $Y_s$ is the target output (ground truth) for sample $s$.
        > * $\hat{Y}_s$ is the predicted output from the model's output node.
        > * $h_j$ is the activation output of hidden layer neuron $j$.
        > * $\sigma$ is the Sigmoid activation function.
        > * $z_{\text{hidden},j}$ is the pre-activation net input to hidden neuron $j$.
        > * $z_{\text{output}}$ is the pre-activation net input to the output neuron.
        > * $x_i$ is the input value at feature index $i$.
        > * $w_{ij}$ is the weight connecting input $i$ to hidden neuron $j$.
        > * $b_j$ is the bias term for hidden neuron $j$.
        > * $w_j$ is the weight connecting hidden neuron $j$ to the output neuron.
        > * $b_{\text{output}}$ is the bias term for the output neuron.
        > * $\Delta w_j$ is the weight update value for the weight connecting hidden neuron $j$ to the output neuron.
        > * $\alpha$ is the learning rate parameter controlling the step size of optimization.
        > * $\Delta b_{\text{output}}$ is the bias update value for the output layer.
        > * $\Delta w_{ij}$ is the weight update value for the weight connecting input $i$ to hidden neuron $j$.
        > * $\Delta b_j$ is the bias update value for hidden neuron $j$.
        
    - **Implementation Details:** The engine implements a raw feedforward backpropagation neural network engineered entirely without external machine learning dependencies. The network structures its parameter matrices dynamically with random initialization between $[-1, 1]$ across two input nodes, two hidden neurons, and a single output neuron. Utilizing supervised batch-epoch processing over 10,000 learning iterations, the trainer systematically minimizes global error using a fixed learning rate ($\eta = 0.5$). Weights ($w$) and biases ($b$) are simultaneously adjusted at each step according to the computed localized gradients, ensuring the final network successfully pushes its classification thresholds to isolate the non-linear XOR vectors.
    ![alt text](..\AllImages\Multi-Layer-Perceptron-XOR-Gate-Classifier.jpg)

2. ## Multiclass Iris Classifier Neural Network
    - **Project Goal:** The algorithm learns to classify physical Iris flower samples into one of three distinct biological species (*Iris setosa*, *Iris versicolor*, or *Iris virginica*) based on four continuous anatomical feature measurements extracted from an imported dataset file.
    - **Mathematical Framework:** The system utilizes a fully connected Feedforward Neural Network architecture optimized via supervised Gradient Descent and Backpropagation. The output distribution is normalized into mutually exclusive probabilities using the Softmax activation function. Network errors are evaluated using a categorical cross-entropy derivative formulation. When a pattern propagates through the system, weights and biases are iteratively updated by projecting errors backward layer-by-layer, scaling adjustments by the local sensitivity curve of the hidden nodes. The mathematical update rule for minimizing the network error matrix entry is governed by the generalized delta rule:
        > $$\text{Loss Function} = E = -\sum_{k} Y_{s,k} \ln(\hat{Y}_{s,k})$$
        >
        > $$\Large h_j = \sigma(z_{\text{hidden},j}) = \sigma\left(\sum_{i} (x_i \cdot w_{ij}) + b_j\right) = \frac{1}{1 + e^{-z_{\text{hidden},j}}}$$
        >
        > $`\hat{Y}_{s,k} = \text{softmax}(z_{\text{output},k}) = \frac{e^{z_{\text{output},k}}}{\sum_{m} e^{z_{\text{output},m}}}`$
        >
        > $$\Large \frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z))$$
        >
        > ---
        >
        > $`\delta_{\text{output},k} = \frac{\partial E}{\partial z_{\text{output},k}} = \hat{Y}_{s,k} - Y_{s,k}`$
        >
        > $$\Large \delta_{\text{hidden},j} = \frac{\partial E}{\partial z_{\text{hidden},j}} = \sum_{k} \left( \frac{\partial E}{\partial z_{\text{output},k}} \cdot \frac{\partial z_{\text{output},k}}{\partial h_j} \right) \cdot \frac{\partial h_j}{\partial z_{\text{hidden},j}}$$
        >
        > $$\Large = \sum_{k} \left( \delta_{\text{output},k} \cdot w_{jk} \right) \cdot h_j(1 - h_j)$$
        >
        > $$\Large = \left( \sum_{k} \delta_{\text{output},k} \cdot w_{jk} \right) \cdot h_j(1 - h_j)$$
        >
        > ---
        >
        > $`\delta_{\text{output},k} = \hat{Y}_{s,k} - Y_{s,k}`$
        >
        > $$\Large \delta_{\text{hidden},j} = \left( \sum_{k} \delta_{\text{output},k} \cdot w_{jk} \right) \cdot h_j(1 - h_j)$$
        >
        > ---
        >
        > $$\Large \Delta w_{jk} = -\alpha \cdot \delta_{\text{output},k} \cdot h_j \qquad \text{and} \qquad \Delta b_{\text{output},k} = -\alpha \cdot \delta_{\text{output},k}$$
        >
        > $$\Large \Delta w_{ij} = -\alpha \cdot \delta_{\text{hidden},j} \cdot x_i \qquad \text{and} \qquad \Delta b_j = -\alpha \cdot \delta_{\text{hidden},j}$$
        >
        > ---
        >
        > * $E$ is the Categorical Cross-Entropy loss for a single training sample.
        > * $Y_{s,k}$ is the target ground truth probability for class $k$ of sample $s$ (`sample.target[i]`).
        > * $\hat{Y}_{s,k}$ is the Softmax predicted probability for class $k$ (`outputNeuron[i]`).
        > * $h_j$ is the activation output of hidden layer neuron $j$ (`hiddenNeuron[i]`).
        > * $\sigma$ is the Sigmoid activation function applied to the hidden layer.
        > * $z_{\text{hidden},j}$ is the pre-activation net input to hidden neuron $j$.
        > * $z_{\text{output},k}$ is the pre-activation net input to output neuron $k$.
        > * $x_i$ is the input value at feature index $i$ (`inputNeuron[i]`).
        > * $w_{ij}$ is the weight connecting input $i$ to hidden neuron $j$ (`WIH[i][j]`).
        > * $b_j$ is the bias term for hidden neuron $j$ (`biasHidden[i]`).
        > * $w_{jk}$ is the weight connecting hidden neuron $j$ to output neuron $k$ (`WHO[j][i]`).
        > * $b_{\text{output},k}$ is the bias term for output neuron $k$ (`biasOutput[i]`).
        > * $\delta_{\text{output},k}$ is the calculated error term for output neuron $k$ (`outputError[i]`).
        > * $\delta_{\text{hidden},j}$ is the calculated gradient/error term for hidden neuron $j$ (`hiddenError[i]`).
        > * $\alpha$ is the learning rate parameter controlling optimization step sizes (`learningRateAlpha`).
        > * $\Delta w_{jk}$ is the update amount subtracted from the hidden-to-output weight matrix.
        > * $\Delta b_{\text{output},k}$ is the update amount subtracted from the output layer bias vector.
        > * $\Delta w_{ij}$ is the update amount subtracted from the input-to-hidden weight matrix.
        > * $\Delta b_j$ is the update amount subtracted from the hidden layer bias vector.
    - **Implementation Details:** The engine compiles a structured 3-layer multilayer perceptron object (`4` inputs $\rightarrow$ `14` hidden units $\rightarrow$ `3` outputs). The structural training pipeline processes an input CSV source dataset containing 150 instances, parsing flower dimensions into a customized data struct object (`IrisSample`). Target labels are mapped directly onto orthogonal One-Hot encoded arrays. To prevent model saturation and weight bias loops, the data collection is mixed uniformly before processing using a stochastic pseudo-random engine shuffle. The runtime network trains across `500` continuous processing epochs using stochastic in-place matrix operations, continuously tuning structural weight layers (`WIH`, `WHO`) and threshold bias arrays until output probabilities consistently match raw flower categories.