import numpy as np
import math

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    """
    Implement for default intermediate activation.
        - call function
        - input gradients
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, x) -> Tensor:
        """TODO: Leaky ReLu forward propagation! """
        return Tensor(np.where(x > 0, x, self.alpha * x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        TODO: Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
        """
        x = self.input_dict["x"]
        return [Tensor(np.where(x > 0, 1, self.alpha))]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    """
    Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients 
    """ 
    
    def call(self, x) -> Tensor:
        return Tensor(1 / (1 + np.exp( - 1 * x)))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
         """
        return [Tensor(output * (1 - output)) for output in self.outputs]
    
    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    """
    Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients
    """

    def call(self, x):
        """Softmax forward propagation!"""
        ## all entries to prevent overflow/underflow issues
        x = x - np.amax(x)
        exps = np.exp(x)
        return exps/np.sum(exps, axis=-1, keepdims=True)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        softmax_output = self.outputs[0]
        # compute outer product to build jacobian matrix, we do np.array because we must loop through 3D output
        gradients = np.array([-np.outer(output, output) for output in softmax_output])
        diag = softmax_output * (1-softmax_output)
        np.einsum('ijj->ij', gradients)[...] = diag
        return [Tensor(gradients)]


