import pytest
import torch
import torch.nn as nn
from transformer_lib.optimizers import AdamW


class TestAdamW:
    """Test cases for AdamW optimizer"""
    
    @pytest.fixture
    def simple_model(self):
        """Simple model for testing optimizer"""
        return nn.Linear(10, 5)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        return x, y
    
    def test_adamw_initialization(self, simple_model):
        """Test AdamW optimizer initializes correctly"""
        optimizer = AdamW(simple_model.parameters())
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
        assert optimizer.param_groups[0]['eps'] == 1e-8
        assert optimizer.param_groups[0]['weight_decay'] == 1e-2
    
    def test_adamw_custom_parameters(self, simple_model):
        """Test AdamW with custom parameters"""
        lr = 0.001
        betas = (0.95, 0.99)
        eps = 1e-6
        weight_decay = 0.01
        
        optimizer = AdamW(
            simple_model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        assert optimizer.param_groups[0]['lr'] == lr
        assert optimizer.param_groups[0]['betas'] == betas
        assert optimizer.param_groups[0]['eps'] == eps
        assert optimizer.param_groups[0]['weight_decay'] == weight_decay
    
    def test_adamw_invalid_parameters(self, simple_model):
        """Test AdamW raises errors for invalid parameters"""
        # Invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            AdamW(simple_model.parameters(), lr=-0.1)
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="Invalid epsilon value"):
            AdamW(simple_model.parameters(), eps=-1e-8)
        
        # Invalid beta1
        with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
            AdamW(simple_model.parameters(), betas=(-0.1, 0.999))
        
        # Invalid beta2
        with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
            AdamW(simple_model.parameters(), betas=(0.9, 1.1))
        
        # Invalid weight decay
        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            AdamW(simple_model.parameters(), weight_decay=-0.1)
    
    def test_adamw_step_basic(self, simple_model, sample_data):
        """Test basic AdamW optimization step"""
        optimizer = AdamW(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        x, y = sample_data
        
        # Store initial parameters
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        # Forward pass
        output = simple_model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        for initial_param, current_param in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial_param, current_param)
    
    def test_adamw_state_initialization(self, simple_model, sample_data):
        """Test AdamW state initialization"""
        optimizer = AdamW(simple_model.parameters())
        criterion = nn.MSELoss()
        
        x, y = sample_data
        
        # Initially, optimizer state should be empty
        assert len(optimizer.state) == 0
        
        # After first step, state should be initialized
        output = simple_model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check state for each parameter
        for param in simple_model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                assert 'step' in state
                assert 'm' in state
                assert 'v' in state
                assert state['step'] == 1
                assert state['m'].shape == param.shape
                assert state['v'].shape == param.shape
    
    def test_adamw_bias_correction(self, simple_model, sample_data):
        """Test AdamW bias correction mechanism"""
        optimizer = AdamW(simple_model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        x, y = sample_data
        
        # Run multiple steps to test bias correction
        for step in range(5):
            output = simple_model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check that step counter increases
            for param in simple_model.parameters():
                if param.grad is not None:
                    assert optimizer.state[param]['step'] == step + 1
    
    def test_adamw_weight_decay(self):
        """Test AdamW weight decay functionality"""
        # Create a simple parameter
        param = nn.Parameter(torch.ones(5, 5))
        optimizer = AdamW([param], lr=0.1, weight_decay=0.1)
        
        # Create dummy gradient
        param.grad = torch.zeros_like(param)
        
        # Store initial parameter values
        initial_param = param.clone()
        
        # Take optimization step
        optimizer.step()
        
        # Parameter should decrease due to weight decay (even with zero gradient)
        assert torch.all(param < initial_param)
    
    def test_adamw_zero_grad(self, simple_model, sample_data):
        """Test AdamW zero_grad functionality"""
        optimizer = AdamW(simple_model.parameters())
        criterion = nn.MSELoss()
        
        x, y = sample_data
        
        # Forward and backward pass
        output = simple_model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check gradients exist
        for param in simple_model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradients are zeroed
        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_adamw_multiple_param_groups(self):
        """Test AdamW with multiple parameter groups"""
        model1 = nn.Linear(5, 3)
        model2 = nn.Linear(3, 1)
        
        optimizer = AdamW([
            {'params': model1.parameters(), 'lr': 0.01},
            {'params': model2.parameters(), 'lr': 0.001}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[1]['lr'] == 0.001
    
    def test_adamw_no_grad_parameters(self, simple_model, sample_data):
        """Test AdamW handles parameters without gradients"""
        optimizer = AdamW(simple_model.parameters())
        criterion = nn.MSELoss()
        
        x, y = sample_data
        
        # Manually set one parameter to not require gradients
        for param in simple_model.parameters():
            param.requires_grad = False
            break
        
        output = simple_model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Should not raise an error
        optimizer.step()
    
    def test_adamw_convergence_simple_problem(self):
        """Test AdamW can solve a simple optimization problem"""
        # Simple quadratic function: f(x) = (x - 2)^2
        target = 2.0
        x = nn.Parameter(torch.tensor([0.0]))
        optimizer = AdamW([x], lr=0.1)
        
        initial_distance = abs(x.item() - target)
        
        # Optimize for several steps
        for _ in range(100):
            loss = (x - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_distance = abs(x.item() - target)
        
        # Should converge closer to target
        assert final_distance < initial_distance
        assert final_distance < 0.1  # Should be quite close
    
    def test_adamw_momentum_behavior(self):
        """Test AdamW momentum accumulation"""
        # Create parameter with consistent gradient direction
        param = nn.Parameter(torch.tensor([1.0]))
        optimizer = AdamW([param], lr=0.01, betas=(0.9, 0.999))
        
        # Apply consistent gradient multiple times
        for _ in range(5):
            param.grad = torch.tensor([1.0])  # Consistent gradient direction
            optimizer.step()
        
        # Check that momentum is being accumulated
        state = optimizer.state[param]
        assert abs(state['m'].item()) > 0.1  # Should have accumulated momentum
    
    def test_adamw_different_tensor_shapes(self):
        """Test AdamW with different parameter shapes"""
        # Different shaped parameters
        params = [
            nn.Parameter(torch.randn(1)),      # 1D
            nn.Parameter(torch.randn(3, 4)),   # 2D
            nn.Parameter(torch.randn(2, 3, 4)) # 3D
        ]
        
        optimizer = AdamW(params)
        
        # Set gradients and step
        for param in params:
            param.grad = torch.randn_like(param)
        
        # Should handle all shapes without error
        optimizer.step()
        
        # Check state is created for all parameters
        for param in params:
            state = optimizer.state[param]
            assert state['m'].shape == param.shape
            assert state['v'].shape == param.shape
    
    def test_adamw_numerical_stability(self):
        """Test AdamW numerical stability with extreme values"""
        # Very small parameter
        param_small = nn.Parameter(torch.tensor([1e-8]))
        optimizer_small = AdamW([param_small], eps=1e-10)
        
        param_small.grad = torch.tensor([1e-8])
        optimizer_small.step()
        
        # Should not produce NaN or inf
        assert torch.isfinite(param_small).all()
        
        # Very large parameter
        param_large = nn.Parameter(torch.tensor([1e8]))
        optimizer_large = AdamW([param_large])
        
        param_large.grad = torch.tensor([1e8])
        optimizer_large.step()
        
        # Should not produce NaN or inf
        assert torch.isfinite(param_large).all()