import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    """
    TODO:
        - call function
        - input_gradients
    Identical to HW1!
    """

    def call(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse = np.mean((y_true-y_pred)**2)
        return Tensor(mse)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Return in the order defined in the call function.
        """
        n = np.product(np.shape(self.input_dict["y_pred"]))
        true_gradient = np.zeros_like(self.input_dict["y_true"])
        pred_gradient = (-2 / n) * (self.input_dict["y_true"] - self.input_dict["y_pred"])
        return [Tensor(pred_gradient), Tensor(true_gradient)]


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1 - eps)


class CategoricalCrossentropy(Loss):
    """
    TODO: Implement CategoricalCrossentropy class
        - call function
        - input_gradients

        Hint: Use clip_0_1 to stabilize calculations
    """

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        length = len(y_true)
        value = -1 / length * np.sum(y_true * np.log(clip_0_1(y_pred)))
        return Tensor(value)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        # gradient for y true is just zero because it is constant
        pred_gradient = (self.input_dict["y_true"] * (-1/clip_0_1(self.input_dict["y_pred"])))
        true_gradient = np.zeros_like(self.input_dict["y_true"])
        return [Tensor(pred_gradient), Tensor(true_gradient)]
     