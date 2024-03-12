from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        ...
        # queue has the output tensors, can check if null
        while(queue):
            current_tensor_id = id(queue.pop(0))
            curr_layer = self.previous_layers[current_tensor_id]
            if curr_layer is not None:
                input_gradients = curr_layer.compose_input_gradients(grads[current_tensor_id])
                for input, grad in zip(curr_layer.inputs, input_gradients):
                    grads[id(input)] = [grad]
                
                weight_gradients = curr_layer.compose_weight_gradients(grads[current_tensor_id])
                for weight, grad in zip(curr_layer.weights, weight_gradients):
                    grads[id(weight)] = [grad]
                    
                for input in curr_layer.inputs:
                    queue.append(input)

        ## Retrieve the sources and make sure that all of the sources have been reached
        out_grads = [grads[id(source)][0] for source in sources]

        disconnected = [f"var{i}" for i, grad in enumerate(out_grads) if grad is None]

        if disconnected:
            print(f"Warning: The following tensors are disconnected from the target graph: {disconnected}")

        return out_grads