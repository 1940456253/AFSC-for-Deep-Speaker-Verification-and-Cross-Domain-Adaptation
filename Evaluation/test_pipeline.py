"""Run with python -m unittest Evaluation.test_pipeline -v."""
import unittest
import numpy as np
import torch
from Features.afsc import TrainableAFSCFilter
from Features.frontend import build_extractor
from Model.system import SpeakerSystem, angular_margin_loss
from Evaluation.metrics import compute_metrics


class PipelineTests(unittest.TestCase):
    def test_afsc_matches_scalar_formula_and_gradient(self):
        bank = TrainableAFSCFilter().double()
        pts = bank.get_bin_points().detach().numpy()
        actual = bank.create_filterbank(torch.device('cpu'), torch.float64)
        expected = np.zeros((80, 257))
        for j in range(80):
            left, centre, right = pts[j:j + 3]
            for k in range(257):
                if left <= k < centre:
                    expected[j, k] = (k - left) / ((centre - left) ** 2 + 1e-12)
                elif centre <= k <= right:
                    expected[j, k] = (right - k) / ((right - centre) ** 2 + 1e-12)
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-10, atol=1e-10)
        self.assertTrue(np.all(np.diff(pts[:-1]) > 0))
        self.assertAlmostEqual(pts[-1], pts[-2], places=8)
        torch.manual_seed(7)
        spec = torch.rand(2, 15, 257, dtype=torch.float64)
        y = bank(spec)
        self.assertLess(float(y.mean(1).abs().max()), 1e-10)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(bank.raw_gaps.grad).all())
        self.assertGreater(float(bank.raw_gaps.grad.abs().sum()), 0)
        # Compare autograd with finite differences away from support boundaries.
        analytic = bank.raw_gaps.grad[20].item()
        original = bank.raw_gaps[20].item()
        with torch.no_grad():
            bank.raw_gaps[20] = original + 1e-5
            plus = bank(spec).square().mean().item()
            bank.raw_gaps[20] = original - 1e-5
            minus = bank(spec).square().mean().item()
            bank.raw_gaps[20] = original
        self.assertAlmostEqual(analytic, (plus - minus) / 2e-5, places=6)

    def test_frontend_shapes(self):
        wav = torch.randn(16000)
        for name, dim in [('afsc', 257), ('fbank', 80), ('mfcc', 80)]:
            result = build_extractor(name)(wav)
            self.assertEqual(result.shape, (98, dim))
            self.assertTrue(torch.isfinite(result).all())

    def test_freeze_and_encoder_backward(self):
        cfg = {'feature': 'afsc', 'embedding_dim': 24, 'channels': [32, 32, 32, 32, 96]}
        model = SpeakerSystem(cfg)
        model.freeze_frontend(True)
        original = model.frontend.raw_gaps.detach().clone()
        out = model(torch.rand(2, 25, 257))
        out.square().mean().backward()
        self.assertIsNone(model.frontend.raw_gaps.grad)
        self.assertTrue(torch.equal(original, model.frontend.raw_gaps))
        self.assertIsNotNone(next(model.encoder.parameters()).grad)

    def test_metrics_operating_points(self):
        self.assertEqual(compute_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])['eer_percent'], 0)
        self.assertEqual(compute_metrics([1, 1, 1, 1], [1, 0, 1, 0])['eer_percent'], 50)
        self.assertEqual(compute_metrics([1, 1, 1, 1], [1, 0, 1, 0])['min_dcf'], 1)
        self.assertEqual(compute_metrics([0, 1], [1, 0])['eer_percent'], 100)
        with self.assertRaises(ValueError):
            compute_metrics([0.2, 0.3], [1, 1])

    def test_learning_rate_and_margin_schedules(self):
        import yaml
        from Model.train import learning_rate, current_margin
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / 'Model/config.yaml').read_text())
        self.assertAlmostEqual(learning_rate(0, cfg), cfg['min_learning_rate'])
        self.assertAlmostEqual(learning_rate(5, cfg), cfg['learning_rate'])
        self.assertAlmostEqual(learning_rate(10, cfg), cfg['min_learning_rate'])
        self.assertEqual(current_margin(0, cfg), 0.3)
        cfg = yaml.safe_load((root / 'Model/legacy_schedule.yaml').read_text())
        self.assertAlmostEqual(learning_rate(11, cfg), cfg['min_learning_rate'])
        self.assertEqual(current_margin(0, cfg), 0.0)
        self.assertEqual(current_margin(8, cfg), 0.3)

    def test_margin_loss_finite(self):
        x = torch.tensor([[0.9, 0.1], [0.2, 0.8]], requires_grad=True)
        loss = angular_margin_loss(x, torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
