#!/usr/bin/env python3
"""
Test MeZO convergence on a simple optimization problem
"""

import torch
import numpy as np
from accelerate_mezo import zo_perturb_parameters, zo_forward, zo_step, zo_update, perturb_parameters

def simple_quadratic_loss(params):
    """Simple quadratic loss: f(x) = sum((x - target)^2)"""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x = params[0][1]  # Extract parameter
    return torch.sum((x - target) ** 2)

def test_mezo_convergence():
    """Test if MeZO can minimize a simple quadratic function"""
    print("Testing MeZO convergence on quadratic function")
    print("Target: [1.0, 2.0, 3.0, 4.0, 5.0]")
    print("=" * 50)
    
    # Initialize parameters far from optimum
    device = torch.device("cpu")
    x = torch.zeros(5, requires_grad=True)
    params = [("x", x)]
    
    # Hyperparameters
    eps = 1e-3
    lr = 0.1
    num_steps = 100
    
    # Create mock objects
    class MockModel:
        def __init__(self, params):
            self.params = params
        
        def eval(self):
            pass
        
        def __call__(self, **kwargs):
            loss = simple_quadratic_loss(self.params)
            return type('obj', (object,), {'loss': loss})
    
    class MockScheduler:
        def __init__(self):
            self.lr = 1.0
            
        def step(self):
            pass
            
        def get_last_lr(self):
            return [self.lr]
    
    model = MockModel(params)
    lr_scheduler = MockScheduler()
    
    # Training loop
    losses = []
    for step in range(num_steps):
        # Use zo_step logic
        zo_random_seed = np.random.randint(1000000000)
        
        # Compute losses using perturb_parameters
        perturb_parameters(params, eps, zo_random_seed, scaling_factor=1)
        loss1 = simple_quadratic_loss(params)
        
        perturb_parameters(params, eps, zo_random_seed, scaling_factor=-2)
        loss2 = simple_quadratic_loss(params)
        
        perturb_parameters(params, eps, zo_random_seed, scaling_factor=1)
        
        # Compute gradient
        projected_grad = ((loss1 - loss2) / (2 * eps)).item()
        
        # Update - pass zo_random_seed instead of z_vectors
        zo_update(params, lr_scheduler, projected_grad, zo_random_seed, lr, weight_decay=0)
        
        # Record loss
        current_loss = simple_quadratic_loss(params).item()
        losses.append(current_loss)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {current_loss:.6f}, x = {x.data.numpy()}")
    
    # Check convergence
    print("\n" + "=" * 50)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final parameters: {x.data.numpy()}")
    print(f"Target parameters: [1.0, 2.0, 3.0, 4.0, 5.0]")
    print(f"Parameter error: {torch.norm(x.data - torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])).item():.6f}")
    
    # Success if loss decreased significantly
    converged = losses[-1] < losses[0] * 0.1 and losses[-1] < 1.0
    if converged:
        print("\n✅ MeZO successfully converged!")
    else:
        print("\n❌ MeZO failed to converge")
    
    return converged

if __name__ == "__main__":
    import sys
    success = test_mezo_convergence()
    sys.exit(0 if success else 1)
