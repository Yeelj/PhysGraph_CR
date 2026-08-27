import unittest

import numpy as np
# Isaac Gym requires its extension to be imported before torch in this project.
from isaacgym import gymapi  # noqa: F401
import torch
import trimesh

from main.dataset.oakink2_dataset_utils import extract_red_black_regions
from main.dataset.semantic_contact import (
    ACTUAL_CONTACT,
    BLACK_PROXIMITY,
    BLACK_VIOLATION,
    CURRICULUM_WEIGHT,
    EXPECTED_CONTACT,
    FORCE_NORMALIZED,
    RED_CONTACT,
    RED_PROXIMITY,
    semantic_curriculum_weight,
    semantic_reward_terms,
)
from lib.rl.network_builder_transformer_bih_graph_improve_correct import (
    LATENT_ENCODER,
    ManoTopology,
    PhysicallyGroundedBias,
    SemanticAdaLNZero,
    SemanticConditionEncoder,
    TransformerLayer,
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

    def test_semantic_condition_keeps_relations_separate_and_excludes_physical_channels(self):
        semantic = torch.zeros(1, 5, 8)
        semantic[:, :, ACTUAL_CONTACT] = 0.9
        semantic[:, :, EXPECTED_CONTACT] = 0.8
        semantic[:, :, RED_PROXIMITY] = 0.25
        semantic[:, :, RED_CONTACT] = 0.6
        semantic[:, :, BLACK_PROXIMITY] = 0.4
        semantic[:, :, BLACK_VIOLATION] = 0.2
        semantic[:, :, FORCE_NORMALIZED] = 0.7
        semantic[:, :, CURRICULUM_WEIGHT] = 0.5

        features = SemanticConditionEncoder.build_relation_features(semantic)

        expected = torch.tensor([0.8, 0.2, 0.6, 0.4, 0.2, 0.5])
        self.assertTrue(torch.allclose(features[0, 0], expected))
        self.assertEqual(features.shape, (1, 5, 6))

        # Actual contact and force stay on the independent physical path in A1.
        semantic[:, :, ACTUAL_CONTACT] = 0.0
        semantic[:, :, FORCE_NORMALIZED] = 0.0
        self.assertTrue(
            torch.allclose(
                features,
                SemanticConditionEncoder.build_relation_features(semantic),
            )
        )

    def test_dynamic_physical_edge_is_independent_of_semantic_contact(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        with torch.no_grad():
            bias.edge_bias_scale.fill_(1.0)
            bias.edge_encoder.weight.zero_()
            bias.edge_encoder.weight[2, 4] = 0.4
        physical_contact = torch.zeros(2, 5)
        physical_contact[1] = 1.0

        output = bias(
            2,
            torch.device("cpu"),
            semantic_contact=None,
            physical_contact=physical_contact,
        )

        node, tool, head = 26, 56, 4
        self.assertEqual(output[0, head, node, tool].item(), 0.0)
        self.assertAlmostEqual(output[1, head, node, tool].item(), 0.4, places=6)
        self.assertAlmostEqual(output[1, head, tool, node].item(), 0.4, places=6)

    def test_explicit_physical_contact_takes_precedence_over_semantic_fallback(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        with torch.no_grad():
            bias.edge_bias_scale.fill_(1.0)
            bias.edge_encoder.weight.zero_()
            bias.edge_encoder.weight[2, 4] = 0.4
        semantic = torch.zeros(1, 5, 8)
        semantic[:, :, 0] = 1.0
        physical_contact = torch.zeros(1, 5)

        output = bias(
            1,
            torch.device("cpu"),
            semantic_contact=semantic,
            physical_contact=physical_contact,
        )

        self.assertEqual(output[0, 4, 26, 56].item(), 0.0)

    def test_attention_bias_is_independent_of_semantic_relations(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        semantic = torch.zeros(2, 5, 8)
        semantic[1, :, EXPECTED_CONTACT:BLACK_VIOLATION + 1] = 1.0
        semantic[1, :, CURRICULUM_WEIGHT] = 1.0
        physical_contact = torch.zeros(2, 5)
        x_pos = torch.zeros(2, bias.topo.num_phys_nodes, 3)
        x_pos[:, bias.TOOL_NODE_ID, 0] = 0.1

        output = bias(
            2,
            torch.device("cpu"),
            semantic_contact=semantic,
            physical_contact=physical_contact,
            x_pos=x_pos,
        )

        self.assertTrue(torch.allclose(output[0], output[1]))
        self.assertFalse(bias.semantic_relation_scale.requires_grad)

    def test_adaln_zero_is_exactly_baseline_preserving(self):
        adaln = SemanticAdaLNZero(d_model=8)
        normalized = torch.randn(2, 7, 8)
        condition = torch.randn(2, 7, 8)
        mask = torch.zeros(2, 7, 1)
        mask[:, 2:4] = 1.0

        conditioned, residual_gate = adaln(normalized, condition, mask)

        self.assertTrue(torch.equal(conditioned, normalized))
        self.assertTrue(torch.equal(residual_gate, torch.ones_like(residual_gate)))
        self.assertTrue(torch.all(adaln.modulation.weight == 0.0))
        self.assertTrue(torch.all(adaln.modulation.bias == 0.0))

    def test_adaln_mask_blocks_non_contact_tokens_after_learning(self):
        adaln = SemanticAdaLNZero(d_model=4)
        with torch.no_grad():
            adaln.modulation.bias[:4].fill_(1.0)
            adaln.modulation.bias[4:8].fill_(2.0)
            adaln.modulation.bias[8:].fill_(3.0)
        normalized = torch.randn(1, 3, 4)
        condition = torch.randn(1, 3, 4)
        mask = torch.tensor([[[0.0], [1.0], [0.0]]])

        conditioned, residual_gate = adaln(normalized, condition, mask)

        self.assertTrue(torch.equal(conditioned[:, 0], normalized[:, 0]))
        self.assertTrue(torch.equal(conditioned[:, 2], normalized[:, 2]))
        self.assertFalse(torch.equal(conditioned[:, 1], normalized[:, 1]))
        self.assertEqual(residual_gate[0, 0, 0].item(), 1.0)
        self.assertGreater(residual_gate[0, 1, 0].item(), 1.0)
        self.assertEqual(residual_gate[0, 2, 0].item(), 1.0)

    def test_zero_initialized_semantic_layer_matches_plain_transformer(self):
        torch.manual_seed(7)
        plain = TransformerLayer(
            d_model=8, nhead=2, dim_feedforward=16, use_semantic_adaln=False
        )
        conditioned_layer = TransformerLayer(
            d_model=8, nhead=2, dim_feedforward=16, use_semantic_adaln=True
        )
        conditioned_layer.load_state_dict(plain.state_dict(), strict=False)
        plain.eval()
        conditioned_layer.eval()
        tokens = torch.randn(2, 6, 8)
        condition = torch.randn(2, 6, 8)
        mask = torch.zeros(2, 6, 1)
        mask[:, 3] = 1.0

        plain_output = plain(tokens)
        conditioned_output = conditioned_layer(
            tokens,
            semantic_condition=condition,
            semantic_token_mask=mask,
        )

        self.assertTrue(torch.equal(plain_output, conditioned_output))

    def test_geometric_bias_is_restricted_to_side_specific_contact_object_pairs(self):
        bias = PhysicallyGroundedBias(ManoTopology(), num_heads=8)
        with torch.no_grad():
            bias.geo_bias_scale.fill_(1.0)
            bias.w_geo.weight.zero_()
            bias.w_geo.bias.fill_(1.0)
        x_pos = torch.zeros(1, bias.topo.num_phys_nodes, 3)

        output = bias(1, torch.device("cpu"), x_pos=x_pos)

        rh_node = bias.rh_contact_node_ids[0].item()
        lh_node = bias.lh_contact_node_ids[0].item()
        tool = bias.TOOL_NODE_ID
        obj = bias.OBJECT_NODE_ID

        self.assertEqual(output[0, 0, rh_node, tool].item(), 1.0)
        self.assertEqual(output[0, 0, tool, rh_node].item(), 1.0)
        self.assertEqual(output[0, 0, lh_node, obj].item(), 1.0)
        self.assertEqual(output[0, 0, obj, lh_node].item(), 1.0)
        self.assertTrue(torch.all(output[0, :, rh_node, tool] == 1.0))
        self.assertTrue(torch.all(output[0, :, lh_node, obj] == 1.0))

        # Cross-side and finger-finger pairs must not receive geometric bias.
        self.assertEqual(output[0, 0, rh_node, obj].item(), 0.0)
        self.assertEqual(output[0, 0, lh_node, tool].item(), 0.0)
        self.assertEqual(
            output[0, 0, bias.rh_contact_node_ids[0], bias.rh_contact_node_ids[1]].item(),
            0.0,
        )
        self.assertEqual(bias.geometric_pair_mask.sum().item(), 20.0)

    def test_latent_encoder_maps_only_rh_contact_tokens_and_last_two_layers(self):
        encoder = LATENT_ENCODER(num_global_layers=4)
        encoder.eval()
        self.assertEqual(
            [layer.semantic_adaln is not None for layer in encoder.layers],
            [False, False, True, True],
        )
        self.assertEqual(
            encoder.rh_contact_token_ids.tolist(),
            [28, 6, 11, 21, 16],
        )

        tokens = torch.zeros(2, 61, encoder.d_model)
        semantic = torch.zeros(2, 5, 8)
        semantic[:, :, EXPECTED_CONTACT] = 1.0
        semantic[:, :, CURRICULUM_WEIGHT] = 1.0
        token_condition, token_mask = encoder.build_semantic_token_condition(
            semantic, tokens
        )

        self.assertEqual(token_condition.shape, tokens.shape)
        self.assertEqual(token_mask.sum().item(), 10.0)
        self.assertTrue(torch.all(token_mask[:, 0] == 0.0))  # policy token
        self.assertTrue(torch.all(token_mask[:, 59:] == 0.0))  # tool and object
        self.assertTrue(
            torch.all(token_mask[:, encoder.rh_contact_token_ids] == 1.0)
        )
        zero_token_ids = torch.where(token_mask[0, :, 0] == 0.0)[0]
        self.assertTrue(torch.all(token_condition[:, zero_token_ids] == 0.0))

        semantic[:, :, CURRICULUM_WEIGHT] = 0.0
        _, pure_imitation_mask = encoder.build_semantic_token_condition(
            semantic, tokens
        )
        self.assertTrue(torch.all(pure_imitation_mask == 0.0))

    def test_semantic_disabled_keeps_plain_transformer_path(self):
        encoder = LATENT_ENCODER(num_global_layers=4)
        tokens = torch.randn(2, 61, encoder.d_model)

        condition, token_mask = encoder.build_semantic_token_condition(None, tokens)

        self.assertIsNone(condition)
        self.assertIsNone(token_mask)

    def test_full_encoder_forward_is_baseline_preserving_and_adaln_is_trainable(self):
        torch.manual_seed(11)
        encoder = LATENT_ENCODER(num_global_layers=4)
        encoder.eval()
        batch_size = 2
        common_obs = {
            "proprioception": torch.randn(batch_size, 158),
            "privileged": torch.randn(batch_size, 118),
            "target": torch.randn(batch_size, 900),
            "phys_node_pos": torch.randn(batch_size, 58, 3),
            "physical_contact": torch.zeros(batch_size, 5),
        }
        semantic_off = torch.zeros(batch_size, 5, 8)
        semantic_on = torch.zeros(batch_size, 5, 8)
        semantic_on[:, :, EXPECTED_CONTACT:BLACK_VIOLATION + 1] = 1.0
        semantic_on[:, :, CURRICULUM_WEIGHT] = 1.0

        output_off = encoder({**common_obs, "semantic_contact": semantic_off})
        output_on = encoder({**common_obs, "semantic_contact": semantic_on})

        self.assertTrue(torch.equal(output_off, output_on))
        output_on.square().sum().backward()
        for layer in encoder.layers[-2:]:
            self.assertIsNotNone(layer.semantic_adaln.modulation.weight.grad)
            self.assertGreater(
                layer.semantic_adaln.modulation.weight.grad.abs().sum().item(), 0.0
            )


if __name__ == "__main__":
    unittest.main()
