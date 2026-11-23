"""B6 parameter selection, sharing, update behavior, and reload checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.train_two_block_clip import B6Model, selected_visual_parameters


class SmallVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.randn(4, 512) * 0.01)
    def forward(self, image):
        x = image
        for block in self.transformer.resblocks:
            x = x + block(x).tanh()
        return self.ln_post(x) @ self.proj


class B6Tests(unittest.TestCase):
    def test_two_blocks_selected_and_block_nine_frozen(self):
        visual = SmallVisual()
        names = [name for name, _ in selected_visual_parameters(visual)]
        self.assertTrue(any(name.startswith('transformer.resblocks.10.') for name in names))
        self.assertTrue(any(name.startswith('transformer.resblocks.11.') for name in names))
        self.assertFalse(any(name.startswith('transformer.resblocks.9.') for name in names))
        self.assertIn('ln_post.weight', names)
        self.assertIn('proj', names)

    def test_shared_visual_update_freeze_and_checkpoint_reload(self):
        torch.manual_seed(539)
        visual = SmallVisual()
        model = B6Model(visual, np.zeros(512, np.float32), np.ones(512, np.float32),
                        np.zeros(512, np.float32), np.ones(512, np.float32))
        self.assertIs(model.visual, visual)
        self.assertFalse(any(p.requires_grad for p in model.visual.parameters()))
        before = {n: p.detach().clone() for n, p in visual.named_parameters()}
        model.set_clip_trainable(True)
        selected = dict(selected_visual_parameters(visual))
        self.assertEqual(set(selected), {n for n, p in visual.named_parameters() if p.requires_grad})
        original, crop = torch.randn(3, 4), torch.randn(3, 4)
        calls = []
        hook = visual.register_forward_hook(lambda *args: calls.append(True))
        output = model(original, crop, True)
        hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(output.shape), (3, 5))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output >= 0).all())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        output.mean().backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in selected.items()))
        for name, parameter in visual.named_parameters():
            if name not in selected:
                self.assertTrue(torch.equal(before[name], parameter), name)
        model.eval()
        expected = model(original, crop, True).detach()
        restored = B6Model(SmallVisual(), np.zeros(512, np.float32), np.ones(512, np.float32),
                           np.zeros(512, np.float32), np.ones(512, np.float32))
        restored.load_state_dict(model.state_dict()); restored.eval()
        torch.testing.assert_close(restored(original, crop, True), expected)
        model.set_clip_trainable(False)
        self.assertFalse(any(p.requires_grad for p in visual.parameters()))


if __name__ == '__main__':
    unittest.main()
