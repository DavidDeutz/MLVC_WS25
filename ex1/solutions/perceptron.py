import numpy as np
from tqdm import trange

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# forward(), fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################


class Perceptron:
    """
    A Perceptron is a type of artificial neural network used in supervised learning.
    It is used to classify input vectors into two classes, based on a linear function.
    """

    def __init__(self, lr=0.5, epochs=100):
        """
        Initializes the Perceptron object.

        Parameters:
        lr (float): Learning rate for the Perceptron. Default is 0.5.
        epochs (int): Number of iterations for training the Perceptron. Default is 100.
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None

    def forward(self, X):
        """
        Perceptron function.

        Parameters:
        X (numpy.ndarray): Input vectors.

        Returns:
        numpy.ndarray: Class labels.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        if self.w is None:
            raise ValueError("Weights are not initialized.\n")

        output = np.dot(self.w, X)  # dot product of weights and inputs
        return np.where(output >= 0, 1, -1)  # step activation function
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X, y):
        """
        Trains the Perceptron on the given input vectors and labels.

        Parameters:
        X (numpy.ndarray): Input vectors.
        y (numpy.ndarray): Labels/target.

        Returns:
        list: Number of misclassified examples at every iteration.
        """
        # X --> Inputs.
        # y --> labels/target.
        # lr --> learning rate.
        # epochs --> Number of iterations.

        X = np.column_stack([X, np.ones(X.shape[0])])

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Initialize weights with zero
        self.w = np.zeros(X.shape[1])

        # Empty list to store how many examples were
        # misclassified at every iteration.
        miss_classifications = []

        # Training.
        for epoch in trange(self.epochs):

            predictions = self.forward(X.T)
            error = y - predictions
            wrong_prediction_indices = np.nonzero(error)[0]

            # print(predictions)
            if (predictions == 0).all():
                print(f"No errors after {epoch} epochs. Training successful!")
            else:
                # sample one wrong prediction at random
                wrong_prediction_idx = np.random.choice(wrong_prediction_indices)
                prediction_for_update = self.forward(X[wrong_prediction_idx, :])
                # update the weights of the perceptron at random
                # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
                self.w += (
                    self.lr
                    * (y[wrong_prediction_idx] - prediction_for_update)
                    * X[wrong_prediction_idx, :]
                )
                # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            # Appending number of misclassified examples
            # at every iteration.

            miss_classifications.append(predictions.shape[0] - np.sum(error == 0))

        return miss_classifications

    def predict(self, X):
        """
        Predicts the class labels for the given input vectors.

        Parameters:
        X (numpy.ndarray): Input vectors.

        Returns:
        numpy.ndarray: Predicted class labels.
        """

        X = np.column_stack([X, np.ones(X.shape[0])])

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        return self.forward(X.T)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
