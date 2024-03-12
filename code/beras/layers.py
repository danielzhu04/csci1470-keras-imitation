import numpy as np
import math

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):
    """
    This class represents a Dense (or Fully Connected) layer.

    """

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]: 
        """

        returns: the weights (and biases) of this Dense Layer
        """
        return [self.w, self.b]

    def call(self, x: Tensor) -> Tensor:
        """
        x: input data of shape [num_samples, input_size]
        returns: the forward pass of the dense layer performed on x
        """
        # x is (num samples, input_size); w is (input_size, output_size), b is (1, output_size)
        return Tensor(np.matmul(x, self.w) + self.b)
        

    def get_input_gradients(self) -> list[Tensor]:
        """
        Return the gradient of this layer with respect to its input, as a list

        returns: a list of gradients in the same order as its inputs
        """
        return [Tensor(self.w)]

    def get_weight_gradients(self) -> list[Tensor]:
        """
        Return the gradients of this layer with respect to its weights (and biases).

        returns: a list of gradients (returned in the same order as self.weights())
        """
        return [Tensor(np.expand_dims(self.input_dict["x"], axis=-1)), Tensor(np.ones_like(self.b))]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        return the initialized weight, bias Variables as according to the initializer.

        initializer: string representing which initializer to use. see below for details
        input_size: size of latent dimension of input
        output_size: size of latent dimension of output
        returns: weight, bias as **Variable**s.

        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        io_size = (input_size, output_size)

        initialize_biases = np.zeros((1, output_size)) #maybe just output size

        b = Tensor(initialize_biases) # initialize biases to output_size x 1 zeros
        initialize_weights = np.zeros(io_size)
        if (initializer == "normal"):
            initialize_weights = np.random.normal(1, size=io_size)
        elif (initializer == "xavier"):
            initialize_weights = np.random.normal(0, math.sqrt(2 / (input_size + output_size)), size=io_size)
        elif (initializer == "kaiming"):
            initialize_weights = np.random.normal(0, math.sqrt(2 / input_size), size=io_size)
        w = Tensor(initialize_weights)
        return w, b