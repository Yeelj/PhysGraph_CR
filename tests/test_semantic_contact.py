import math
import unittest

import numpy as np
# Isaac Gym requires its extension to be imported before torch in this project.
from isaacgym import gymapi  # noqa: F401
import torch
import trimesh

from main.dataset.oakink2_dataset_utils import extract_red_black_regions
from main.dataset.semantic_contact import semantic_curriculum_weight, semantic_reward_terms
from lib.rl.network_builder_transformer_bih_graph_improve_correct import (
    ManoTopology,
    PhysicallyGroundedBias,
)


class SemanticContactTest(unittest.TestCase):
    def test_color_regions_are_independent_and_fixed_size(self):
        mesh = trimesh.creation.box()
        colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        colors[:, 3] = 255
        colors[:4, 0] = 255
        mesh.visual.vertex_colors = colors
        regions = extract_red_black_regions(mesh, point_count=7)
        self.assertEqual(regions["tool_red_points"].shape, (7, 3))
        self.assertEqual(regions["tool_black_points"].shape, (7, 3))
        self.assertEqual(regions["tool_red_vertex_count"], 4)
        self.assertEqual(regions["tool_black_vertex_count"], 4)

    def test_imitation_dominant_curriculum_boundaries(self):
        total = 1000
        values = [
            semantic_curriculum_weight(step, total).item()
            for step in (0, 50, 300, 600, 1000)
        ]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 0.0)
        self.assertAlmostEqual(values[2], 0.25, places=6)
        self.assertAlmostEqual(values[3], 1.0, places=6)
        self.assertAlmostEqual(values[4], 1.0, places=6)

    def test_reward_red_good_black_bad_and_missing(self):
        actual = torch.tensor([[1.0, 0.0]])
        expected = torch.tensor([[1.0, 1.0]])
        red = torch.tensor([[1.0, 0.0]])
        black = torch.tensor([[0.0, 0.0]])
        force = torch.tensor([[2.0, 0.0]])
        red_reward, black_penalty, missing_penalty, force_penalty, raw = semantic_reward_terms(
            actual, expected, red, black, force, force_safe=12.0
        )
        self.assertAlmostEqual(red_reward.item(), 0.5, places=6)
        self.assertEqual(black_penalty.item(), 0.0)
        self.assertAlmostEqual(missing_penalty.item(), 0.5, places=6)
        self.assertEqual(force_penalty.item(), 0.0)
        self.assertAlmostEqual(raw.item(), 0.4, places=6)

    def test_physical_and_semantic_relations_coexist_per_pair_and_per_batch(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        with torch.no_grad():
            bias.edge_bias_scale.fill_(1.0)
            bias.edge_encoder.weight.zero_()
            bias.edge_encoder.weight[2, 4] = 0.4
            bias.semantic_relation_scale.zero_()
            bias.semantic_relation_scale[bias.EXPECTED_REL, 4] = math.atanh(0.25)

        semantic = torch.zeros(2, 5, 8)
        semantic[:, :, 0] = 1.0  # physical contact is always active
        semantic[1, :, 1] = 1.0  # expected relation only in batch item 1
        semantic[1, :, 7] = 1.0  # full semantic curriculum only in item 1
        output = bias(2, torch.device("cpu"), semantic_contact=semantic)
        node, tool, head = 26, 56, 4
        self.assertAlmostEqual(output[0, head, node, tool].item(), 0.4, places=6)
        self.assertGreater(output[1, head, node, tool].item(), 0.4)
        semantic_only = output[1, head, node, tool] - output[0, head, node, tool]
        self.assertLessEqual(abs(semantic_only.item()), 2.0)

    def test_relation_signs_and_head_roles(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        semantic = torch.zeros(1, 5, 8)
        semantic[:, :, 1:6] = 1.0
        semantic[:, :, 7] = 1.0
        values = bias._semantic_relation_values(bias._semantic_strengths(semantic))
        self.assertTrue(torch.all(values[:, 4:6] > 0))
        self.assertTrue(torch.all(values[:, 6:8] < 0))
        self.assertTrue(torch.all(values[:, :4] == 0))


if __name__ == "__main__":
    unittest.main()
