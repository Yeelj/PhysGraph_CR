"""CPU policy checks: exercise real observation layouts without a GPU simulator."""

import unittest

from isaacgym import gymapi  # noqa: F401; import the Isaac extension before torch
import torch

from main.dexhand_spec import ObservationLayout, get_hand_spec
from main.dataset.semantic_contact import CURRICULUM_WEIGHT, EXPECTED_CONTACT
from lib.rl.network_builder_transformer_bih_graph_improve_correct import (
    BimanualHandToolPolicy,
    LATENT_ENCODER,
)
from lib.rl.res_models import ModelA2CContinuousLogStdBiH


def make_policy(name="artimano", future_length=1):
    spec = get_hand_spec(name)
    return BimanualHandToolPolicy(
        {"dexhand": name, "obs_future_length": future_length, "compile_encoder": False},
        actions_num=2 * (6 + spec.n_dofs),
    )


def make_observation(name, future_length=1, semantic=False):
    spec = get_hand_spec(name)
    layout = ObservationLayout(spec, future_length=future_length)
    observation = {
        "proprioception": torch.randn(2, 2 * layout.hand_proprioception_dim),
        "privileged": torch.randn(2, 2 * layout.hand_privileged_dim),
        "target": torch.randn(2, 2 * layout.hand_target_dim),
        "phys_node_pos": torch.randn(2, 2 * spec.n_bodies + 2, 3),
        "physical_contact": torch.rand(2, 5),
    }
    if semantic:
        observation["semantic_contact"] = torch.rand(2, 5, 8)
        observation["semantic_contact"][:, :, EXPECTED_CONTACT] = 1.0
        observation["semantic_contact"][:, :, CURRICULUM_WEIGHT] = 1.0
    return observation


def wrap_policy(policy):
    return ModelA2CContinuousLogStdBiH.Network(
        policy,
        obs_shape={}, normalize_value=False, normalize_input=False, value_size=1,
    )


class MultiHandPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_four_hands_forward_backward_with_semantics_enabled_and_disabled(self):
        for name, expected_actions in (("artimano", 56), ("allegro", 44), ("inspire", 36), ("shadow", 56)):
            for semantic in (False, True):
                with self.subTest(hand=name, semantic=semantic):
                    policy = make_policy(name)
                    observation = make_observation(name, semantic=semantic)
                    mu, logstd, value, _, base_action = policy(observation)
                    self.assertEqual(mu.shape, (2, expected_actions))
                    self.assertEqual(logstd.shape, (expected_actions,))
                    self.assertEqual(value.shape, (2, 1))
                    self.assertIsNone(base_action)
                    (mu.square().mean() + value.square().mean() + logstd.mean()).backward()
                    for parameter in (policy.sigma, policy.action_miu_head.net[-1].weight, policy.value_head.net[-1].weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    if semantic:
                        for layer in policy.latent_encoder.layers[-2:]:
                            self.assertGreater(layer.semantic_adaln.modulation.weight.grad.abs().sum().item(), 0.0)

    def test_multiple_future_frames_use_field_major_targets_and_complete_bps(self):
        for name in ("artimano", "allegro", "inspire", "shadow"):
            with self.subTest(hand=name):
                policy = make_policy(name, future_length=2)
                observation = make_observation(name, future_length=2)
                encoder = policy.latent_encoder
                layout, spec = encoder.layout, encoder.rh_spec
                target = observation["target"][:, :layout.hand_target_dim]
                # Distinct frame/link/coordinate values catch accidental mixing
                # of the flattened field-major future arrays.
                expected = torch.arange(2 * (spec.n_bodies - 1) * 3).reshape(1, 2, spec.n_bodies - 1, 3).expand(2, -1, -1, -1).float()
                for field, offset in (("delta_joints_pos", 0), ("joints_vel", 1000), ("delta_joints_vel", 2000)):
                    target[:, layout.target_slices[field]] = (expected + offset).reshape(2, -1)
                _, _, links, object_obs = encoder.unpack_hand_observation(
                    observation["proprioception"][:, :layout.hand_proprioception_dim],
                    observation["privileged"][:, :layout.hand_privileged_dim], target,
                )
                self.assertTrue(torch.equal(links[:, :, :3], expected[:, 0]))
                self.assertTrue(torch.equal(links[:, :, 3:6], expected[:, 0] + 1000))
                self.assertTrue(torch.equal(links[:, :, 9:12], expected[:, 1]))
                self.assertTrue(torch.equal(object_obs[:, 1:], target[:, layout.target_slices["bps"]]))
                mu, _, value, _, _ = policy(observation)
                (mu.sum() + value.sum()).backward()
                self.assertTrue(torch.isfinite(mu).all())

    def test_allegro_shared_finger_conditions_are_averaged_and_contact_is_maximum(self):
        encoder = LATENT_ENCODER(dexhand="allegro")
        encoder.eval()
        token_ids = encoder.rh_contact_token_ids.tolist()
        self.assertEqual(len(set(token_ids)), 4)
        duplicate_id = next(node for node in token_ids if token_ids.count(node) > 1)
        slots = [i for i, node in enumerate(token_ids) if node == duplicate_id]
        semantic = torch.rand(2, 5, 8)
        semantic[:, :, CURRICULUM_WEIGHT] = 1.0
        tokens = torch.zeros(2, encoder.physical_token_node_ids.numel() + 1, 64)
        condition, mask = encoder.build_semantic_token_condition(semantic, tokens)
        expected = encoder.semantic_condition_encoder(semantic)[:, slots].mean(dim=1)
        self.assertTrue(torch.allclose(condition[:, duplicate_id], expected))
        self.assertTrue(torch.equal(mask[:, duplicate_id], torch.ones(2, 1)))
        self.assertEqual(mask.sum().item(), 8.0)

        bias = encoder.bias_gen
        with torch.no_grad():
            bias.edge_bias_scale.fill_(1.0)
            bias.edge_encoder.weight.zero_()
            bias.edge_encoder.weight[2, 0] = 1.0
        physical = torch.zeros(2, 5)
        physical[:, slots[0]] = 0.3
        physical[:, slots[1]] = 0.7
        output = bias(2, torch.device("cpu"), physical_contact=physical)
        body_id = int(bias.rh_contact_node_ids[slots[0]])
        self.assertTrue(torch.allclose(output[:, 0, body_id, bias.TOOL_NODE_ID], torch.full((2,), 0.7)))

    def test_checkpoint_roundtrip_and_cross_hand_rejection_including_equal_dimensions(self):
        for name in ("artimano", "allegro", "inspire", "shadow"):
            with self.subTest(hand=name):
                original, restored = wrap_policy(make_policy(name)), wrap_policy(make_policy(name))
                restored.load_state_dict(original.state_dict())
                observation = make_observation(name)
                original.eval()
                restored.eval()
                self.assertTrue(torch.equal(original.a2c_network(observation)[0], restored.a2c_network(observation)[0]))
        for source, destination in (("artimano", "shadow"), ("inspire", "allegro")):
            with self.subTest(source=source, destination=destination):
                with self.assertRaisesRegex(RuntimeError, "Checkpoint hand/DOF/body ordering"):
                    wrap_policy(make_policy(destination)).load_state_dict(wrap_policy(make_policy(source)).state_dict())

    def test_legacy_artimano_checkpoint_and_compiled_prefix_migration(self):
        original = wrap_policy(make_policy())
        legacy = {
            key.replace("latent_encoder.", "latent_encoder._orig_mod."): value
            for key, value in original.state_dict().items()
            if not key.endswith("_extra_state")
            and "semantic_condition_encoder." not in key
            and ".semantic_adaln." not in key
        }
        restored = wrap_policy(make_policy())
        restored.load_state_dict(legacy)
        observation = make_observation("artimano")
        original.eval()
        restored.eval()
        self.assertTrue(torch.equal(original.a2c_network(observation)[0], restored.a2c_network(observation)[0]))
        metadata_free_shadow = {key: value for key, value in make_policy("shadow").state_dict().items() if not key.endswith("_extra_state")}
        with self.assertRaisesRegex(RuntimeError, "no hand metadata"):
            make_policy("shadow").load_state_dict(metadata_free_shadow, strict=False)

    def test_checkpoint_metadata_is_independent_and_checks_joint_order(self):
        policy = make_policy("shadow")
        checkpoint = policy.state_dict()
        dofs = checkpoint["_extra_state"]["rh_hand"]["dof_names"]
        dofs[0], dofs[1] = dofs[1], dofs[0]
        with self.assertRaisesRegex(RuntimeError, "Checkpoint hand/DOF/body ordering"):
            policy.load_state_dict(checkpoint)

    def test_invalid_shapes_and_excluded_hand_fail_early(self):
        with self.assertRaises((KeyError, ValueError)):
            make_policy("xhand")
        with self.assertRaisesRegex(ValueError, "requires 44 actions"):
            BimanualHandToolPolicy({"dexhand": "allegro", "compile_encoder": False}, actions_num=56)
        observation = make_observation("inspire")
        observation["target"] = torch.zeros(2, 900)
        with self.assertRaisesRegex(ValueError, "target must have shape"):
            make_policy("inspire")(observation)


if __name__ == "__main__":
    unittest.main()
