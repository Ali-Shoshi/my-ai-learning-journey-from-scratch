# Decision Tree

1. ## Decision Tree Binary Beach Classifie
    - **Project Goal:** The algorithm learns to recursively partition a categorical dataset of environmental observations into pure sub-segments to predict whether an individual will visit the beach based on structural information gain.
    - **Mathematical Framework:** The model is a parametric classification tree that uses Gini Impurity to evaluate the quality of categorical splits. At any given node with $C$ unique target classes, where $p_i$ represents the probability of a sample belonging to class $i$, the Gini Impurity $I_G$ is calculated as:
        > $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
        >
        > $$I_G(\text{Split}) = \frac{N_{\text{left}}}{N_{\text{total}}} I_G(\text{left}) + \frac{N_{\text{right}}}{N_{\text{total}}} I_G(\text{right})$$
    - **Implementation Details:** This system implements a binary Decision Tree classifier completely from scratch in C++. The training pipeline processes a categorical dataset containing environmental attributes (Weather, Temperature, and Weekend flags) mapped to a binary target class (goToBeach). During execution, the engine iteratively scans every feature branch inside findBestSplit using independent lexical scoping blocks to compute regional Gini indices. The dataset is recursively divided into left and right subset vectors to construct an explicit pointer-based tree structure (Node*) until max depth or total class purity is reached, enabling clean non-linear decision boundary printing via structured console indentation logs.
    
        |Weather | Temperature | Weekend | Will Go to Beach? |
        | :--- | :--- | :--- | :--- |
        | Sunny | Hot | Yes | Yes |
        | Sunny | Hot | No | Yes |
        | Rainy | Cool | Yes | No |
        | Sunny | Cool | Yes | No |
        | Rainy | Hot | No | No |
        | Sunny | Hot | Yes | Yes |
        | Sunny | Hot | No | Yes |

        ![alt text](1000000096.jpg)
2. ## Decision Tree Regression Project
    - **Goal:** The algorithm learns to map a continuous 1D feature variable ($x$) to a continuous target scalar ($y$) by partitioning the numerical input space into localized, uniform regions via supervised recursive binary splitting.
    - **Mathematical Framework:** The model optimizes split boundaries by maximizing variance reduction ($\Delta \sigma^2$) at each internal node. Given a parent dataset $D$ split into a left child $D_L$ and a right child $D_R$ via a threshold $s$, the optimal parameter satisfies:
        >$$\Large \Delta \sigma^2 = \sigma^2(D) - \left( \frac{|D_L|}{|D|} \sigma^2(D_L) + \frac{|D_R|}{|D|} \sigma^2(D_R) \right)$$
    - **Implementation Details:** It implements a soft-bounded Regression Tree trained from scratch in native C++ using structural pointers and lambda-based sorting frameworks. The pipeline accepts a structured vector of historical input coordinates and iteratively partitions the feature values after ordering them monotonically to evaluate child subsets. At each recursion layer, the system exhaustively scans the midpoint values between neighboring elements, weighting the Mean Squared Error (MSE) equivalents of resulting children against parent node impurity to determine the absolute maximum variance compression. By enforcing strict early stopping thresholds via an absolute depth constraint ($depth \le 3$), minimum partition capacities ($|D| \le 2$), and a convergence tolerance ($10^{-6}$), the model compresses data trends into a binary tree layout of leaf predictions computed as regional targets averages.