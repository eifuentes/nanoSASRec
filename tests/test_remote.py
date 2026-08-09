import unittest
from types import SimpleNamespace

import data
import remote


class RemoteCommandTests(unittest.TestCase):
    def test_base_dataset_mapping(self):
        self.assertEqual(remote.base_dataset_for("kuairand-27k"), "kuairand-27k")
        self.assertEqual(remote.base_dataset_for("kuairand-27k-item555-user5"), "kuairand-27k")
        self.assertEqual(remote.base_dataset_for("kuairand-pure-item1-user5"), "kuairand-pure")
        self.assertEqual(remote.base_dataset_for("ml-20m"), "ml-20m")

    def test_data_py_recipe_slug_round_trips_to_base_dataset(self):
        slug = data.kuairand_recipe_slug(item_floor=555, user_floor=5)
        self.assertEqual(remote.base_dataset_for(f"kuairand-27k-{slug}"), "kuairand-27k")
        self.assertEqual(remote.base_dataset_for(f"kuairand-pure-{slug}"), "kuairand-pure")
        rule_slug = data.kuairand_recipe_slug(
            item_floor=555,
            user_floor=5,
            interaction_rule="long-view-not-hate-tab1",
            exclude_rand=True,
        )
        self.assertEqual(remote.base_dataset_for(f"kuairand-27k-{rule_slug}"), "kuairand-27k")

    def test_volume_data_dir(self):
        self.assertEqual(remote.volume_data_dir("kuairand-27k"), "/volumes/kuairand-27k")

    def test_normalize_sets_missing_data_dir(self):
        cmd = "python -u data.py --dataset kuairand-27k --kuairand-item-floor 2000"
        self.assertEqual(
            remote.normalize_cmd_data_dir(cmd),
            "python -u data.py --dataset kuairand-27k --kuairand-item-floor 2000 --data-dir /volumes/kuairand-27k",
        )

    def test_normalize_maps_recipe_stamped_dataset_to_base_volume(self):
        cmd = (
            "python -u train_v9.py "
            "--dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand --epochs 1"
        )
        self.assertEqual(
            remote.normalize_cmd_data_dir(cmd),
            "python -u train_v9.py --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand --epochs 1 --data-dir /volumes/kuairand-27k",
        )

    def test_normalize_preserves_existing_data_dir(self):
        cmd = "python -u data.py --dataset kuairand-pure --data-dir /tmp/custom"
        self.assertEqual(
            remote.normalize_cmd_data_dir(cmd),
            "python -u data.py --dataset kuairand-pure --data-dir /volumes/kuairand-pure",
        )

    def test_normalize_overrides_relative_data_dir(self):
        cmd = "python -u train_v9.py --dataset ml-20m --data-dir data"
        self.assertEqual(
            remote.normalize_cmd_data_dir(cmd),
            "python -u train_v9.py --dataset ml-20m --data-dir /volumes/ml-20m",
        )

    def test_normalize_overrides_equals_data_dir(self):
        cmd = "python -u train_v9.py --dataset ml-20m --data-dir=/tmp/custom"
        self.assertEqual(
            remote.normalize_cmd_data_dir(cmd),
            "python -u train_v9.py --dataset ml-20m --data-dir /volumes/ml-20m",
        )

    def test_normalize_without_dataset_is_rejected(self):
        cmd = "python -u data.py --help"
        with self.assertRaisesRegex(ValueError, "must include --dataset"):
            remote.normalize_cmd_data_dir(cmd)

    def test_python_data_command_detection(self):
        self.assertTrue(remote._is_python_data_command(["python", "-u", "data.py", "--help"]))
        self.assertFalse(remote._is_python_data_command(["python", "-u", "train_v9.py"]))

    def test_remote_requires_data_checkpoint_hook(self):
        remote._require_data_stage_callback(data)
        with self.assertRaisesRegex(RuntimeError, "KUAIRAND_STAGE_FILE_CALLBACK"):
            remote._require_data_stage_callback(SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "KUAIRAND_ITEM_COUNT_CACHE_CALLBACK"):
            remote._require_data_stage_callback(
                SimpleNamespace(
                    KUAIRAND_STAGE_FILE_CALLBACK=None,
                    KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK=None,
                )
            )

    def test_gpu_entrypoint_rejects_data_commands(self):
        remote._reject_data_command_for_gpu_entrypoint("python -u train_v9.py --dataset ml-20m")
        with self.assertRaisesRegex(ValueError, "remote.py::prep"):
            remote._reject_data_command_for_gpu_entrypoint("python -u data.py --dataset kuairand-pure")

    def test_selected_volumes_uses_only_inferred_base_dataset(self):
        volumes = {
            "ml-1m": object(),
            "ml-20m": object(),
            "kuairand-pure": object(),
            "kuairand-27k": object(),
        }
        selected = remote._selected_volumes(
            "python -u train_v9.py --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand",
            volumes,
        )
        self.assertEqual(list(selected), ["kuairand-27k"])


if __name__ == "__main__":
    unittest.main()
