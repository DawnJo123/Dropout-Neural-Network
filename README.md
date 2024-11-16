## Overview
This project demonstrates the use of a neural network with **Dropout Regularization** to predict California housing prices based on the California Housing dataset. The focus is on implementing dropout to mitigate overfitting, ensuring the model generalizes better to unseen data.

## Aim
The primary objective of this repository is to illustrate the application of dropout in a neural network for a regression task. By adding dropout layers, we aim to:

1. Prevent the model from overfitting to the training data.
2. Enhance the model's ability to generalize to new data.
3. Compare the effectiveness of dropout on the loss trends and test error.

## Features
- **Data Preprocessing**: Standardization of input features for efficient training.
- **Dropout Regularization**: Implementation of dropout in hidden layers to prevent overfitting.
- **Visualization**: Loss trends plotted to observe training performance.
- **Evaluation**: Calculation of Mean Squared Error (MSE) for model evaluation.

## Requirements
Ensure you have the following installed:
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the dependencies using:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Dataset
The **California Housing dataset** is used in this project. It contains information about various features of houses in California and their corresponding median house prices. This dataset is fetched using the `sklearn.datasets.fetch_california_housing` API.

## Code Structure
1. **Data Preparation**  
   - The dataset is loaded and split into training and testing sets.
   - Features are scaled using `StandardScaler` for better convergence during training.

2. **Neural Network Architecture**  
   - A fully connected neural network with two hidden layers.
   - Dropout layers are added after each hidden layer to randomly deactivate neurons during training.

3. **Training Loop**  
   - The model is trained for 100 epochs using the **Adam optimizer** and **Mean Squared Error (MSE)** loss function.
   - Loss values are logged for visualization.

4. **Evaluation**  
   - Predictions are made on the test set.
   - The performance is measured using Mean Squared Error (MSE).

5. **Visualization**  
   - A plot of loss values over epochs is generated to observe the training dynamics.

## Neural Network with Dropout
### Architecture
- **Input Layer**: Accepts 8 features corresponding to the dataset.
- **Hidden Layer 1**: 25 neurons with ReLU activation and dropout.
- **Hidden Layer 2**: 30 neurons with ReLU activation and dropout.
- **Output Layer**: 1 neuron for predicting housing prices.

### Dropout Regularization
- Dropout prevents the model from relying too heavily on any single neuron.
- During training, neurons are randomly set to zero with a probability defined in the dropout layer (default: 0.5).

## Training Details
- **Optimizer**: Adam optimizer with a learning rate of 0.01.
- **Loss Function**: Mean Squared Error (MSE).
- **Epochs**: 100.

## Results
### Loss Plot
A plot showing the loss decreasing over 100 epochs is generated, highlighting the model's training progress.

## Observations
- Dropout significantly reduces the risk of overfitting, as seen in the training loss trend.
- The MSE on the test set indicates the model's ability to generalize to new data.

## Conclusion
This repository highlights the importance of dropout in training neural networks for regression tasks. By applying dropout, the model achieves better generalization and reduced overfitting, making it suitable for real-world data.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions are welcome! Feel free to fork the repository, add new features, or enhance the implementation. Submit pull requests with detailed descriptions of your changes.
