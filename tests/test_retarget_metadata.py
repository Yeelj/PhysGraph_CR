import unittest
import warnings

import numpy as np

from main.dexhand_spec import get_hand_spec
from main.retarget_metadata import normalize_retarget_data


class RetargetMetadataTest(unittest.TestCase):
    @staticmethod
    def trajectory(hand_name="artimano", side="right"):
        spec = get_hand_spec(hand_name, side)
        return {
            "dexhand": hand_name,
            "side": side,
            "dof_names": list(spec.dof_names),
            "body_names": list(spec.body_names),
            "opt_dof_pos": np.arange(3 * len(spec.dof_names), dtype=np.float32).reshape(3, -1),
            "opt_joints_pos": np.zeros((3, len(spec.body_names), 3), dtype=np.float32),
        }

    def test_explicit_actor_order_reorders_by_names_without_mutating_input(self):
        for hand_name in ("artimano", "allegro", "inspire", "shadow"):
            for side in ("right", "left"):
                with self.subTest(hand=hand_name, side=side):
                    payload = self.trajectory(hand_name, side)
                    canonical = tuple(payload["dof_names"])
                    expected = payload["opt_dof_pos"].copy()
                    payload["dof_names"] = list(reversed(canonical))
                    payload["opt_dof_pos"] = expected[:, ::-1]
                    normalized = normalize_retarget_data(
                        payload, hand_name, side, target_dof_names=canonical
                    )
                    np.testing.assert_array_equal(normalized["opt_dof_pos"], expected)
                    self.assertEqual(normalized["dof_names"], list(canonical))
                    self.assertEqual(payload["dof_names"], list(reversed(canonical)))

    def test_offline_fk_keeps_declared_column_order(self):
        payload = self.trajectory("inspire")
        payload["dof_names"].reverse()
        normalized = normalize_retarget_data(payload, "inspire", "rh")
        self.assertEqual(normalized["dof_names"], payload["dof_names"])
        np.testing.assert_array_equal(normalized["opt_dof_pos"], payload["opt_dof_pos"])

    def test_legacy_orders_match_original_exporter_for_all_assets(self):
        for hand_name in ("artimano", "allegro", "inspire", "shadow"):
            for side in ("right", "left"):
                with self.subTest(hand=hand_name, side=side):
                    payload = self.trajectory(hand_name, side)
                    for key in ("dexhand", "side", "dof_names", "body_names"):
                        payload.pop(key)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        normalized = normalize_retarget_data(payload, hand_name, side)
                    self.assertTrue(any("Legacy retarget" in str(item.message) for item in caught))
                    self.assertEqual(normalized["dof_names"], list(get_hand_spec(hand_name, side).dof_names))
                    np.testing.assert_array_equal(normalized["opt_dof_pos"], payload["opt_dof_pos"])

    def test_same_dimension_different_hand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dexhand"):
            normalize_retarget_data(self.trajectory("artimano"), "shadow", "right")

    def test_wrong_side_is_rejected_even_for_same_joint_names(self):
        with self.assertRaisesRegex(ValueError, "side"):
            normalize_retarget_data(self.trajectory("shadow"), "shadow", "left")

    def test_known_legacy_directory_prevents_cross_hand_or_side_reuse(self):
        payload = self.trajectory("artimano")
        for key in ("dexhand", "side", "dof_names", "body_names"):
            payload.pop(key)
        for path in (
            "data/retargeting/TACO/mano2shadow_rh/sequence.pkl",
            "data/retargeting/OakInk-v2/mano2artimano_lh/sequence.pkl",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "directory"):
                normalize_retarget_data(payload, "artimano", "rh", source_path=path)

    def test_incorrect_names_shape_and_finite_values_are_rejected(self):
        wrong_names = self.trajectory()
        wrong_names["dof_names"][0] = "incorrect_joint"
        with self.assertRaisesRegex(ValueError, "dof_names"):
            normalize_retarget_data(wrong_names, "artimano", "right")
        wrong_shape = self.trajectory()
        wrong_shape["opt_dof_pos"] = wrong_shape["opt_dof_pos"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            normalize_retarget_data(wrong_shape, "artimano", "right")
        non_finite = self.trajectory()
        non_finite["opt_dof_pos"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            normalize_retarget_data(non_finite, "artimano", "right")

    def test_correction_metadata_identity_is_checked_as_well(self):
        payload = self.trajectory()
        payload["contact_retarget"] = {"dexhand": "shadow", "side": "right"}
        with self.assertRaisesRegex(ValueError, "dexhand"):
            normalize_retarget_data(payload, "artimano", "right")

    def test_joint_positions_require_matching_body_metadata(self):
        payload = self.trajectory("allegro")
        payload["opt_joints_pos"] = payload["opt_joints_pos"][:, :-1]
        with self.assertRaisesRegex(ValueError, "opt_joints_pos shape"):
            normalize_retarget_data(payload, "allegro", "right")


if __name__ == "__main__":
    unittest.main()
