"""
Unit Tests for Production Koopman-JEPA

Tests cover:
1. Static initialization
2. Numerical stability
3. Theoretical properties (Theorem 3.4)
4. Forward/backward pass
5. Loss computation stability
"""

import unittest
import torch
import numpy as np

from model import (
    Conv1DEncoder,
    LinearPredictor,
    CryptoKoopmanJEPA,
    create_crypto_feature_tensor
)


class TestConv1DEncoder(unittest.TestCase):
    """Test the convolutional encoder"""
    
    def test_static_initialization(self):
        """Test that encoder initializes feature_size statically"""
        encoder = Conv1DEncoder(
            input_channels=3,
            sequence_length=128,
            latent_dim=16
        )
        
        # Should have feature_size set at init
        self.assertIsNotNone(encoder.feature_size)
        self.assertIsInstance(encoder.feature_size, int)
        self.assertGreater(encoder.feature_size, 0)
        
        # Should have projection layer initialized
        self.assertIsNotNone(encoder.projection)
        self.assertEqual(encoder.projection.in_features, encoder.feature_size)
        self.assertEqual(encoder.projection.out_features, 16)
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        encoder = Conv1DEncoder(
            input_channels=3,
            sequence_length=128,
            latent_dim=16
        )
        
        x = torch.randn(4, 3, 128)
        z = encoder(x)
        
        self.assertEqual(z.shape, (4, 16))
    
    def test_no_dynamic_layer_creation(self):
        """Test that forward pass doesn't create new layers"""
        encoder = Conv1DEncoder(
            input_channels=3,
            sequence_length=128,
            latent_dim=16
        )
        
        # Get layer count before forward
        layers_before = len(list(encoder.parameters()))
        
        # Forward pass
        x = torch.randn(4, 3, 128)
        _ = encoder(x)
        
        # Get layer count after forward
        layers_after = len(list(encoder.parameters()))
        
        # Should be same (no dynamic creation)
        self.assertEqual(layers_before, layers_after)


class TestLinearPredictor(unittest.TestCase):
    """Test the linear predictor"""
    
    def test_identity_initialization(self):
        """Test that predictor initializes as identity"""
        predictor = LinearPredictor(latent_dim=16, init_identity=True)
        
        # M should be close to identity
        I = torch.eye(16)  # noqa: E741
        diff = torch.norm(predictor.M - I, p='fro').item()
        
        self.assertLess(diff, 1e-5, "M should be initialized as identity")
    
    def test_identity_deviation_zero(self):
        """Test that identity_deviation returns ~0 for identity matrix"""
        predictor = LinearPredictor(latent_dim=16, init_identity=True)
        
        deviation = predictor.identity_deviation()
        
        self.assertLess(deviation, 0.01, "Identity deviation should be near 0")
    
    def test_forward_preserves_identity(self):
        """Test that forward pass preserves input when M=I"""
        predictor = LinearPredictor(latent_dim=16, init_identity=True)
        predictor.eval()
        
        with torch.no_grad():
            z = torch.randn(4, 16)
            z_pred = predictor(z)
            
            # Should be approximately equal to input
            diff = torch.norm(z - z_pred, p='fro').item()
            self.assertLess(diff, 0.1, "Output should match input when M=I")


class TestSpectralRegularization(unittest.TestCase):
    """Test the stable spectral regularization"""
    
    def test_spectral_loss_near_zero_for_identity(self):
        """Test that spectral loss is near 0 for identity-initialized M"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear',
            init_identity=True
        )
        
        loss = model.compute_spectral_regularization()
        
        # Should be very small (M*M^T ≈ I when M=I)
        self.assertLess(loss.item(), 0.01, "Spectral loss should be near 0 for M=I")
    
    def test_spectral_loss_differentiable(self):
        """Test that spectral loss has proper gradients"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear',
            init_identity=True
        )
        
        loss = model.compute_spectral_regularization()
        loss.backward()
        
        # Check that M has gradients
        self.assertIsNotNone(model.predictor.M.grad)
        self.assertFalse(torch.isnan(model.predictor.M.grad).any(), "No NaN gradients")
        self.assertFalse(torch.isinf(model.predictor.M.grad).any(), "No inf gradients")
    
    def test_spectral_loss_stable_for_degenerate_eigenvalues(self):
        """Test stability when eigenvalues are degenerate (all near 1)"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear',
            init_identity=True
        )
        
        # M is identity → all eigenvalues = 1 (fully degenerate)
        # This would break torch.linalg.eigvals but our Frobenius norm is stable
        
        for _ in range(5):
            loss = model.compute_spectral_regularization()
            loss.backward()
            
            # Should be stable (no NaN or inf)
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())
            self.assertFalse(torch.isnan(model.predictor.M.grad).any())


class TestRegimeConsistencyLoss(unittest.TestCase):
    """Test the numerically stable KL divergence"""
    
    def test_kl_no_nan_with_zero_probabilities(self):
        """Test that KL divergence doesn't produce NaN even with zero probs"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            num_regimes=8
        )
        
        # Create embeddings that might produce zero probabilities
        z_context = torch.randn(4, 16)
        z_pred = torch.randn(4, 16)
        
        loss = model.compute_regime_consistency_loss(z_context, z_pred)
        
        # Should be finite (no NaN or inf)
        self.assertFalse(torch.isnan(loss).any(), "KL loss should not be NaN")
        self.assertFalse(torch.isinf(loss).any(), "KL loss should not be inf")
        self.assertTrue(torch.isfinite(loss).all(), "KL loss should be finite")
    
    def test_kl_has_gradients(self):
        """Test that KL loss produces proper gradients"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            num_regimes=8
        )
        
        z_context = torch.randn(4, 16, requires_grad=True)
        z_pred = torch.randn(4, 16, requires_grad=True)
        
        loss = model.compute_regime_consistency_loss(z_context, z_pred)
        loss.backward()
        
        # Should have gradients
        self.assertIsNotNone(z_context.grad)
        self.assertIsNotNone(z_pred.grad)
        self.assertFalse(torch.isnan(z_context.grad).any())
        self.assertFalse(torch.isnan(z_pred.grad).any())


class TestFeatureCreation(unittest.TestCase):
    """Test the epsilon-safe feature creation"""
    
    def test_feature_shape(self):
        """Test that features have correct shape"""
        prices = np.random.rand(200).astype(np.float32) * 1000 + 50000
        features = create_crypto_feature_tensor(prices, window_size=128)
        
        self.assertEqual(features.shape, (3, 128))
    
    def test_feature_normalization(self):
        """Test that first channel (normalized prices) has mean≈0, std≈1"""
        prices = np.random.rand(200).astype(np.float32) * 1000 + 50000
        features = create_crypto_feature_tensor(prices, window_size=128)
        
        # First channel is normalized prices
        normalized_prices = features[0, :]
        
        mean = normalized_prices.mean().item()
        std = normalized_prices.std().item()
        
        self.assertAlmostEqual(mean, 0.0, places=5, msg="Mean should be near 0")
        self.assertAlmostEqual(std, 1.0, places=5, msg="Std should be near 1")
    
    def test_no_nan_with_constant_prices(self):
        """Test that constant prices don't produce NaN (epsilon protection)"""
        # Constant prices would cause division by zero without epsilon
        prices = np.full(200, 50000.0, dtype=np.float32)
        features = create_crypto_feature_tensor(prices, window_size=128)
        
        # Should be finite (no NaN from division by zero)
        self.assertFalse(torch.isnan(features).any(), "Features should not contain NaN")
        self.assertTrue(torch.isfinite(features).all(), "All features should be finite")
    
    def test_no_nan_with_zero_prices(self):
        """Test handling of edge case with near-zero prices"""
        prices = np.random.rand(200).astype(np.float32) * 0.01  # Very small prices
        features = create_crypto_feature_tensor(prices, window_size=128)
        
        # Should be finite
        self.assertFalse(torch.isnan(features).any())
        self.assertTrue(torch.isfinite(features).all())


class TestFullModel(unittest.TestCase):
    """Test the complete model"""
    
    def test_forward_pass(self):
        """Test full forward pass"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear'
        )
        
        x_ctx = torch.randn(4, 3, 128)
        x_tgt = torch.randn(4, 3, 128)
        
        outputs = model(x_ctx, x_tgt)
        
        self.assertIn('z_context', outputs)
        self.assertIn('z_pred', outputs)
        self.assertIn('z_target', outputs)
        
        self.assertEqual(outputs['z_context'].shape, (4, 16))
        self.assertEqual(outputs['z_pred'].shape, (4, 16))
        self.assertEqual(outputs['z_target'].shape, (4, 16))
    
    def test_full_loss_computation(self):
        """Test that full loss computation is stable"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear'
        )
        
        x_ctx = torch.randn(4, 3, 128)
        x_tgt = torch.randn(4, 3, 128)
        p_ctx = torch.randn(4, 1, 128).abs() + 100  # Positive prices
        p_tgt = torch.randn(4, 1, 128).abs() + 100
        
        losses = model.compute_koopman_crypto_loss(x_ctx, x_tgt, p_ctx, p_tgt)
        
        # All losses should be finite
        for name, loss in losses.items():
            self.assertFalse(torch.isnan(loss).any(), f"{name} loss should not be NaN")
            self.assertFalse(torch.isinf(loss).any(), f"{name} loss should not be inf")
            self.assertTrue(loss.item() >= 0, f"{name} loss should be non-negative")
    
    def test_backward_pass(self):
        """Test that backward pass is stable"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16,
            predictor_type='linear'
        )
        
        x_ctx = torch.randn(4, 3, 128)
        x_tgt = torch.randn(4, 3, 128)
        p_ctx = torch.randn(4, 1, 128).abs() + 100
        p_tgt = torch.randn(4, 1, 128).abs() + 100
        
        losses = model.compute_koopman_crypto_loss(x_ctx, x_tgt, p_ctx, p_tgt)
        losses['total'].backward()
        
        # Check that encoder has gradients and they're finite
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"{name} should have gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"{name} grad should not be NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"{name} grad should not be inf")
    
    def test_ema_update(self):
        """Test that EMA update works correctly"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16
        )
        
        # Get initial target encoder weights
        target_weights_before = {
            name: param.clone() 
            for name, param in model.target_encoder.named_parameters()
        }
        
        # Do a forward pass and update
        x = torch.randn(4, 3, 128)
        _ = model.encoder(x)
        
        # Manually modify encoder weights
        for param in model.encoder.parameters():
            param.data += 0.1
        
        # Update target encoder
        model.update_target_encoder()
        
        # Check that target encoder weights changed
        for name, param in model.target_encoder.named_parameters():
            old_weight = target_weights_before[name]
            changed = not torch.allclose(param, old_weight)
            self.assertTrue(changed, f"Target encoder {name} should have changed")


class TestRepresentationCollapse(unittest.TestCase):
    """Test variance normalization prevents collapse"""
    
    def test_target_embeddings_have_variance(self):
        """Test that target embeddings have non-zero variance"""
        model = CryptoKoopmanJEPA(
            input_channels=3,
            sequence_length=128,
            latent_dim=16
        )
        model.eval()
        
        with torch.no_grad():
            x_ctx = torch.randn(32, 3, 128)
            x_tgt = torch.randn(32, 3, 128)
            
            outputs = model(x_ctx, x_tgt)
            z_target = outputs['z_target']
            
            # Compute variance per dimension
            var_per_dim = z_target.var(dim=0)
            
            # Should have non-zero variance (not collapsed)
            self.assertTrue((var_per_dim > 0.01).all(), "Embeddings should have variance")
            self.assertFalse(torch.isnan(var_per_dim).any(), "Variance should not be NaN")


def run_tests():
    """Run all tests and report results"""
    print("="*80)
    print("Running Koopman-JEPA Unit Tests")
    print("="*80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConv1DEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralRegularization))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeConsistencyLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestFullModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRepresentationCollapse))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {len(result.failures)} TESTS FAILED")
        print(f"✗ {len(result.errors)} TESTS HAD ERRORS")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
