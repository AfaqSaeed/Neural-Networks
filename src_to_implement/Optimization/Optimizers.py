import numpy as np

class Sgd:
    def __init__(self, learning_rate: float) -> None:
        """
        Constructor method for Stochastic Gradient Descent (SGD).

        Parameters:
        - learning_rate: The learning rate for updating the weights.
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_matrix, gradient_matrix):
        """
        Update the weights using the Stochastic Gradient Descent (SGD) algorithm.

        Parameters:
        - weight_matrix: The current weights of the model.
        - gradient_matrix: The gradient of the loss with respect to the weights.

        Returns:
        - updated_weights: The updated weights after applying the SGD update rule.
        """
        # Applying the SGD update rule to calculate the updated weights
        updated_weights = weight_matrix - (self.learning_rate * gradient_matrix)
        
        return updated_weights
