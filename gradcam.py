import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
            return None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, edge_index, **kwargs):
        self.model.eval()

        # Forward pass
        out = self.model(x, edge_index, **kwargs)

        # Backward pass
        self.model.zero_grad()
        out.backward(torch.ones_like(out))

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=0)
        grad_cam = torch.relu(torch.sum(self.activations * weights, dim=1))

        return grad_cam, self.activations, weights
