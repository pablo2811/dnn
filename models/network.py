import numpy as np

from typing import Optional, TypeVar
from models.util import sigmoid, sigmoid_prime, softmax, softmax, softmax_prime

TFloat = TypeVar('TFloat', float, np.ndarray) 

class Network:
    """Multi-Layer Perceptron, a simple neural network with fully-connected layers alternating with sigmoid activations."""
    def __init__(self, sizes: list[int], loss: str = "mse", normalize: str = None):
        """
        Args:
        - sizes: list of layer widths L_i (lengths of input and activation vectors).
        """
        # Initialize biases and weights with random normal distribution (naively).
        # Weights are indexed by target node first.
        self.num_layers = len(sizes)
        self.sizes = sizes
        # List of (num_layers - 1) vectors of shape (L_1), (L_2), ...
        self.biases = [np.random.randn(n) for n in sizes[1:]]
        # List of (num_layers - 1) vectors of shape (L_i, L_{i-1}).
        self.weights = [np.random.randn(n_target, n_source)
                        for n_source, n_target in zip(sizes[:-1], sizes[1:])]
        self.loss = loss
        self.normalize = normalize

    def feedforward(self, a: np.ndarray, memoize: bool = False) -> np.ndarray:
        """
        Run the network on a batch.

        Args:
        - a: input vector, shape (B, I), dtype float64.
        """
        if memoize:
            fs: list[np.ndarray] = []
            # Values after activation function (including inputs to the first layer), shapes (I), (L_1), ..., (O).
            gs: list[np.ndarray] = [a]
        else: 
            res = a

        i = 0
        for b, w in zip(self.biases, self.weights):
            fs_next = (gs[0] if memoize else gs[-1]) @ w.T  + b
            if memoize:
                fs.append(fs_next)

            if i == self.num_layers - 2 and self.normalize == "softmax":
              gs[0] = softmax(fs_next)
            else:
              gs[0] = sigmoid(fs_next)

            if memoize:
                gs.append(gs[0])
            i += 1

        if memoize:
          return fs, gs

        return gs[0]


    def update_mini_batch(self, x_mini_batch: np.ndarray, y_mini_batch: np.ndarray, eta: float) -> None:
        """
        Make a single step of backpropagation and gradient descent.

        Args:
        - x_mini_batch: shape (B, I) where B is mini_batch_size.
        - y_mini_batch: shape (B, O).
        - eta: learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Accumulate gradients by running backprop one dataitem at a time (without vectorization).
        # for x, y in zip(x_mini_batch, y_mini_batch):
        nabla_b, nabla_w = self.backprop(x_mini_batch, y_mini_batch)

        # Gradient descent step.
        self.weights = [w - nw * (eta / len(x_mini_batch))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - nb * (eta / len(x_mini_batch))
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation for a batch input.

        Args:
        - x: input, shape (B, I)
        - y: target label (one-hot encoded), shape (B, O)

        Returns (nabla_b, nabla_w), where:
        - nabla_b is a list of gradients over biases (shape (L_i)), for each layer.
        - nabla_w is a list of gradients over weights (shape (L_i, L_{i-1})), for each layer.
        """
        # Go forward, remembering all activations.
        # Values before activation function, layer by layer, shapes (L_1), ..., (10).
        fs, gs = self.feedforward(x, True)

        # Now go backward from the final cost applying backpropagation.
        dLdg = self.cost_derivative(gs[-1], y)  # shape initially (10), then layer by layer .
        fs_derivs: list[np.ndarray] = []
        gs_derivs: list[np.ndarray] = [dLdg]

        i = len(fs) - 1
        for w in self.weights[::-1]:
            if i == len(fs) - 1 and self.normalize == "softmax":
                activation_prime = softmax_prime(fs[i])
            else:
                activation_prime = sigmoid_prime(fs[i])

            fs_derivs.append(np.multiply(gs_derivs[-1], activation_prime))
            gs_derivs.append(fs_derivs[-1] @ w)
            i -= 1


        nabla_w = []
        for i in range(len(fs_derivs)):
          nabla_w.append(fs_derivs[len(fs_derivs) - i - 1].T @ gs[i])

        nabla_b = [np.sum(elem, axis=0) for elem in fs_derivs[::-1]]

        return nabla_b, nabla_w

    def cost_derivative(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of loss over output activations.

        Args:
        - a: output activations, shape (B, O).
        - y: target values (one-hot encoded labels), shape (B, O).

        Returns gradients, shape (10).
        """

        if self.loss == "mse":
          return (a - y.astype(np.float64))
        elif self.loss == "log-loss":
          return (a - y.astype(np.float64))/(a * (1 - a))


    def evaluate(self, x_test_data: np.ndarray, y_test_data: np.ndarray) -> float:
        """
        Compute accuracy: the ratio of correct answers for test_data.

        Args:
        - x_test_data: shape (B, I)
        - y_test_data: shape (B, O)
        """
        test_results: list[bool] = []
        output_label = np.argmax(self.feedforward(x_test_data),axis=1)
        target_label = np.argmax(y_test_data, axis=1)
        test_results.append(output_label == target_label)

        return np.mean(test_results)

    def SGD(self,
            training_data: tuple[np.ndarray, np.ndarray],
            epochs: int,
            mini_batch_size: int,
            eta: float,
            test_data: Optional[tuple[np.ndarray, np.ndarray]] = None) -> None:
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for epoch in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                i_begin = i * mini_batch_size
                i_end = (i + 1) * mini_batch_size
                self.update_mini_batch(x_train[i_begin:i_end], y_train[i_begin:i_end], eta)
            if test_data:
                accuracy = self.evaluate(x_test, y_test)
                print(f"Epoch: {epoch}, Accuracy: {accuracy}")
            else:
                print(f"Epoch: {epoch}")