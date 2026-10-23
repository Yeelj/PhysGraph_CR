"""Pure stdlib tests; simulator and ML dependencies are deliberately unnecessary."""

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from main.dexhand_spec import ObservationLayout, SUPPORTED_HANDS, get_hand_spec


ROOT = Path(__file__).resolve().parents[1]


def class_literals(filename, classname):
    tree = ast.parse((ROOT / "physgraph_envs/lib/envs/dexhands" / filename).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    values = {}
    for node in ast.walk(node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                try:
                    values[target.attr] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


class HandSpecTests(unittest.TestCase):
    def test_matches_existing_classes_without_importing_them(self):
        for hand in SUPPORTED_HANDS:
            for side in ("right", "left"):
                with self.subTest(hand=hand, side=side):
                    spec = get_hand_spec(hand, side)
                    filename = "artimano_real.py" if hand == "artimano" else hand + ".py"
                    classname = hand.capitalize()
                    if hand == "allegro":
                        classname += "RH" if side == "right" else "LH"
                    values = class_literals(filename, classname)
                    prefix = ("R_" if side == "right" else "L_") if hand == "inspire" else ""
                    for field in ("body_names", "dof_names", "contact_body_names"):
                        self.assertEqual(getattr(spec, field), tuple(prefix + value for value in values[field]))

    def test_counts_and_tree_connectivity_both_sides(self):
        for hand, counts in {"artimano": (22, 28), "allegro": (16, 21), "inspire": (12, 18), "shadow": (22, 28)}.items():
            for side in ("right", "left"):
                with self.subTest(hand=hand, side=side):
                    spec = get_hand_spec(hand, side)
                    self.assertEqual((spec.n_dofs, spec.n_bodies), counts)
                    self.assertEqual(spec.root_body_id, 0)
                    self.assertEqual(len(spec.graph_edges), spec.n_bodies - 1)
                    children = [child for _, child in spec.graph_edges]
                    self.assertEqual(sorted(children), list(range(1, spec.n_bodies)))
                    reachable = {0}
                    for _ in range(spec.n_bodies):
                        reachable.update(child for parent, child in spec.graph_edges if parent in reachable)
                    self.assertEqual(reachable, set(range(spec.n_bodies)))
                    self.assertEqual(len(spec.contact_body_ids), 5)
                    expected_contacts = 4 if hand == "allegro" else 5
                    self.assertEqual(len(spec.unique_contact_body_ids), expected_contacts)
                    self.assertEqual(len(spec.finger_groups), 5)
                    for slot, group in enumerate(spec.finger_groups):
                        self.assertIn(spec.contact_body_ids[slot], group)
                    self.assertEqual(set().union(*map(set, spec.finger_groups)), set(range(1, spec.n_bodies)))

    def test_preserves_artimano_synergy_groups(self):
        self.assertEqual(get_hand_spec("artimano").synergy_groups, ((5, 10, 15, 20, 27), (1, 6, 11, 16, 21), (3, 8, 13, 18, 24)))

    def test_mimic_joints_are_simulated_dofs(self):
        spec = get_hand_spec("inspire")
        root = ET.parse(spec.urdf_path).getroot()
        mimics = {joint.attrib["name"] for joint in root.findall("joint") if joint.find("mimic") is not None}
        self.assertTrue(mimics)
        self.assertTrue(mimics.issubset(spec.dof_names))

    def test_left_allegro_semantic_slots_mirror_right(self):
        right, left = get_hand_spec("allegro", "right"), get_hand_spec("allegro", "left")
        self.assertEqual(right.contact_body_ids, (9, 4, 14, 19, 19))
        self.assertEqual(left.contact_body_ids, (9, 19, 14, 4, 4))
        self.assertEqual(right.finger_groups[3], right.finger_groups[4])

    def test_aliases_rejection_immutability_and_signatures(self):
        self.assertIs(get_hand_spec("Shadow", "rh"), get_hand_spec("shadow"))
        self.assertIs(get_hand_spec("shadow_lh"), get_hand_spec("shadow", "left"))
        for name in ("xhand", "invalid", ""):
            with self.assertRaises(ValueError):
                get_hand_spec(name)
        with self.assertRaises(ValueError):
            get_hand_spec("artimano", "invalid")
        spec = get_hand_spec("artimano")
        with self.assertRaises(FrozenInstanceError):
            spec.name = "shadow"
        self.assertEqual(json.loads(json.dumps(spec.signature())), spec.signature())
        self.assertEqual(len(spec.signature_hash), 64)
        self.assertNotEqual(spec.signature_hash, get_hand_spec("artimano", "left").signature_hash)
        self.assertNotEqual(spec.signature_hash, get_hand_spec("shadow").signature_hash)


class ObservationLayoutTests(unittest.TestCase):
    def test_dimensions_and_contiguous_field_major_slices(self):
        for hand in SUPPORTED_HANDS:
            spec = get_hand_spec(hand)
            for k in (1, 2, 4):
                with self.subTest(hand=hand, k=k):
                    layout = ObservationLayout(spec, future_length=k)
                    self.assertEqual(layout.proprioception_dim, 3 * spec.n_dofs + 13)
                    self.assertEqual(layout.privileged_dim, spec.n_dofs + 37)
                    self.assertEqual(layout.target_dim, 128 + spec.n_bodies + k * (46 + 9 * (spec.n_bodies - 1) + 5))
                    self.assertEqual(layout.hand_target_dim, layout.target_dim)
                    self.assertEqual(layout.hand_proprioception_dim, layout.proprioception_dim)
                    self.assertEqual(layout.hand_privileged_dim, layout.privileged_dim)
                    self.assertEqual(layout.hand_token_dim, 4 * spec.n_dofs + 49)
                    self.assertEqual(layout.curr_token_dim, 46 * k + spec.n_bodies)
                    self.assertEqual(layout.link_token_dim, 9 * k)
                    end = 0
                    for name, index in layout.target_slices.items():
                        self.assertEqual(index.start, end)
                        end = index.stop
                        shape_size = 1
                        for dim in layout.target_field_shapes[name]:
                            shape_size *= dim
                        self.assertEqual(index.stop - index.start, shape_size)
                    self.assertEqual(end, layout.target_dim)
                    self.assertEqual(layout.target_slices["obj_to_joints"].stop - layout.target_slices["obj_to_joints"].start, spec.n_bodies)
                    self.assertEqual(layout.target_field_shapes["delta_joints_pos"], (k, spec.n_bodies - 1, 3))
                    self.assertEqual(layout.target_field_shapes["gt_tips_distance"], (k, 5))

    def test_original_artimano_target_offsets_unchanged(self):
        layout = ObservationLayout(get_hand_spec("artimano"))
        self.assertEqual(layout.target_dim, 450)
        self.assertEqual(layout.target_slices["delta_joints_pos"], slice(23, 104))
        self.assertEqual(layout.target_slices["delta_manip_obj_pos"], slice(266, 269))
        self.assertEqual(layout.target_slices["obj_to_joints"], slice(289, 317))
        self.assertEqual(layout.target_slices["gt_tips_distance"], slice(317, 322))
        self.assertEqual(layout.target_slices["bps"], slice(322, 450))

    def test_rejects_invalid_future_lengths(self):
        for k in (0, -1, 1.5, True, "1"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                ObservationLayout(get_hand_spec("artimano"), future_length=k)


if __name__ == "__main__":
    unittest.main()
