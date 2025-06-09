import unittest
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

# Attempt to import from accelerate_mezo.py, assuming it's in the same directory or PYTHONPATH
try:
    from accelerate_mezo import zo_perturb_parameters, zo_forward, zo_step, zo_update, perturb_parameters
except ImportError:
    # Fallback for environments where direct import might fail (e.g. CI, specific project structures)
    # This requires accelerate_mezo.py to be discoverable.
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from accelerate_mezo import zo_perturb_parameters, zo_forward, zo_step, zo_update, perturb_parameters


# Helper mock classes
class MockModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights deterministically for reproducibility
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids is used as the direct input to the linear layer
        logits = self.linear(input_ids.float()) # Ensure float input
        
        loss = None
        # Use mean of logits as a proxy for loss if not otherwise computed
        # This ensures the loss is a scalar.
        if logits.numel() > 0:
            loss = logits.mean()
        else: # Handle empty batch case
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=False)

        # Return a dictionary-like object or an object with a .loss attribute
        class Output:
            def __init__(self, loss_val, logits_val):
                self.loss = loss_val
                self.logits = logits_val
        return Output(loss_val=loss, logits_val=logits)

class MockLRScheduler:
    def __init__(self, lr_val=1.0): # Default to 1.0 as it's often a multiplier
        self.lr_val = lr_val
        self._last_lr = [lr_val]

    def get_last_lr(self):
        return self._last_lr

    def step(self):
        pass # No change in LR for this mock

class TestMeZoNumeric(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MockModel(in_features=5, out_features=1).to(self.device)
        self.named_params = list(self.model.named_parameters())
        self.eps = 1e-3
        self.lr = 1e-4
        self.weight_decay = 0.01
        
        # Batch of inputs for the model
        self.inputs = {
            'input_ids': torch.randn(2, 5, device=self.device), # batch_size=2, in_features=5
            # 'labels': torch.randn(2,1, device=self.device) # Optional: if model calculates loss with labels
        }
        # Ensure inputs are on the correct device
        for k, v in self.inputs.items():
            self.inputs[k] = v.to(self.device)

    def assert_tensors_equal(self, t1, t2, msg=None, delta=1e-5):
        self.assertTrue(torch.allclose(t1, t2, atol=delta), msg=f"{msg}\nTensor 1: {t1}\nTensor 2: {t2}")

    def test_zo_perturb_parameters(self):
        original_params_data = [p.data.clone() for _, p in self.named_params]
        
        # Generate z_vectors
        torch.manual_seed(42) # For reproducibility of z_vectors
        z_vectors = [torch.randn_like(p.data) for _, p in self.named_params]

        # Test positive perturbation
        zo_perturb_parameters(self.named_params, self.eps, scaling_factor=1, z_vectors=z_vectors)
        for i, (_, param) in enumerate(self.named_params):
            expected_param = original_params_data[i] + self.eps * z_vectors[i]
            self.assert_tensors_equal(param.data, expected_param, "Positive perturbation failed")

        # Restore params for next test (by perturbing back)
        zo_perturb_parameters(self.named_params, self.eps, scaling_factor=-1, z_vectors=z_vectors)
        for i, (_, param) in enumerate(self.named_params):
             self.assert_tensors_equal(param.data, original_params_data[i], "Restoration after positive perturbation failed")

        # Test negative perturbation (from original)
        zo_perturb_parameters(self.named_params, self.eps, scaling_factor=-1, z_vectors=z_vectors)
        for i, (_, param) in enumerate(self.named_params):
            expected_param = original_params_data[i] - self.eps * z_vectors[i]
            self.assert_tensors_equal(param.data, expected_param, "Negative perturbation failed")
        
        # Restore params to original
        zo_perturb_parameters(self.named_params, self.eps, scaling_factor=1, z_vectors=z_vectors)


    def test_zo_step_parameter_restoration_and_gradient(self):
        original_params_data = [p.data.clone() for _, p in self.model.named_parameters()]
        
        # Set a specific numpy seed because zo_step uses np.random.randint for its internal torch generator seed
        np.random.seed(123)
        loss1_step, projected_grad_step, zo_random_seed = zo_step(
            self.model, self.inputs, self.named_params, self.eps, self.device
        )

        # 1. Test parameter restoration
        for i, (_, param) in enumerate(self.named_params):
            self.assert_tensors_equal(
                param.data, original_params_data[i], f"Parameter {self.named_params[i][0]} not restored"
            )

        # 2. Verify loss1 and loss2 (implicitly, by recalculating them)
        # Create a copy of the model to manually perturb
        temp_model_plus = deepcopy(self.model)
        named_params_plus = list(temp_model_plus.named_parameters())
        for i, (_, param) in enumerate(named_params_plus): # Ensure original state for perturbation
            param.data.copy_(original_params_data[i])
        
        # Use the same random seed to generate the same perturbations
        perturb_parameters(named_params_plus, self.eps, zo_random_seed, scaling_factor=1)
        manual_loss1 = zo_forward(temp_model_plus, self.inputs)
        self.assertAlmostEqual(loss1_step.item(), manual_loss1.item(), delta=1e-5, msg="loss1 from zo_step mismatch")

        temp_model_minus = deepcopy(self.model)
        named_params_minus = list(temp_model_minus.named_parameters())
        for i, (_, param) in enumerate(named_params_minus): # Ensure original state for perturbation
            param.data.copy_(original_params_data[i])

        # Perturb by -eps (using the same seed)
        perturb_parameters(named_params_minus, self.eps, zo_random_seed, scaling_factor=-1)
        manual_loss2 = zo_forward(temp_model_minus, self.inputs)
        
        # The projected_grad is (loss1 - loss2) / (2 * eps)
        expected_projected_grad = (manual_loss1 - manual_loss2) / (2 * self.eps)
        self.assertAlmostEqual(projected_grad_step, expected_projected_grad.item(), delta=1e-5, msg="Projected gradient calculation mismatch")


    def test_zo_update(self):
        original_params_data = {name: p.data.clone() for name, p in self.named_params}
        
        # Use a fixed projected gradient and random seed for deterministic test
        projected_grad = 0.5 
        zo_random_seed = 42  # Use an integer seed instead of z_vectors
        
        # Generate the z_vectors that would be created with this seed for verification
        torch.manual_seed(zo_random_seed)
        z_vectors = [torch.randn_like(p.data) for _, p in self.named_params]
        
        # Reset parameters to original state before update
        for (name, param), original in zip(self.named_params, original_params_data.values()):
            param.data.copy_(original)
        
        mock_scheduler = MockLRScheduler(lr_val=1.0) # Scheduler returns a factor of 1.0

        zo_update(self.named_params, mock_scheduler, projected_grad, zo_random_seed, self.lr, self.weight_decay)

        current_lr = mock_scheduler.get_last_lr()[0] * self.lr

        for name, param in self.named_params:
            original_param = original_params_data[name]
            z_vec = z_vectors[[n for n, _ in self.named_params].index(name)] # Get corresponding z vector

            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                # Apply gradient and weight decay
                expected_param = original_param * (1 - current_lr * self.weight_decay) - current_lr * projected_grad * z_vec
            else:
                # Only apply gradient
                expected_param = original_param - current_lr * projected_grad * z_vec
            
            self.assert_tensors_equal(param.data, expected_param, f"Parameter {name} update failed", delta=1e-6)

if __name__ == '__main__':
    unittest.main()
