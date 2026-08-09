import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import data

KUAIRAND_HEADER = [
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "is_hate",
    "long_view",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_rand",
    "tab",
    "play_time_ms",
    "duration_ms",
]


def write_kuairand_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KUAIRAND_HEADER)
        writer.writeheader()
        for row in rows:
            complete = {column: 0 for column in KUAIRAND_HEADER}
            complete.update(row)
            writer.writerow(complete)


class KuaiRandArchiveTests(unittest.TestCase):
    def test_verify_md5_accepts_expected_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "archive.bin")
            path.write_bytes(b"kuairand")
            data.verify_md5(str(path), hashlib.md5(b"kuairand").hexdigest())
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                data.verify_md5(str(path), "0" * 32)

    def test_selective_extraction_flattens_known_member_and_ignores_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp, "sample.tar.gz")
            output = Path(tmp, "logs")
            with tarfile.open(archive_path, "w:gz") as archive:
                wanted = b"user_id,video_id\n1,2\n"
                wanted_info = tarfile.TarInfo("KuaiRand/data/wanted.csv")
                wanted_info.size = len(wanted)
                archive.addfile(wanted_info, io.BytesIO(wanted))

                unwanted = b"nope"
                unsafe_info = tarfile.TarInfo("../../outside.txt")
                unsafe_info.size = len(unwanted)
                archive.addfile(unsafe_info, io.BytesIO(unwanted))

            data.extract_selected_tar(str(archive_path), str(output), ("wanted.csv",))
            self.assertEqual((output / "wanted.csv").read_bytes(), wanted)
            self.assertFalse(Path(tmp, "outside.txt").exists())

    def test_selective_extraction_reports_missing_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp, "sample.tar.gz")
            with tarfile.open(archive_path, "w:gz"):
                pass
            with self.assertRaisesRegex(FileNotFoundError, "missing required log files"):
                data.extract_selected_tar(str(archive_path), str(Path(tmp, "logs")), ("missing.csv",))


class KuaiRandDuckDBPipelineTests(unittest.TestCase):
    def test_fingerprints_change_on_expected_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(csv_path, [])
            raw = data._kuairand_raw_parquet_fingerprint([str(csv_path)], "pure")
            self.assertEqual(len(raw), 64)
            stage = data._kuairand_stage_fingerprint(raw, "all-impressions", False, False)
            self.assertEqual(stage, data._kuairand_stage_fingerprint(raw, "all-impressions", False, False))
            self.assertNotEqual(stage, data._kuairand_stage_fingerprint(raw, "long-view-not-hate", False, False))
            self.assertNotEqual(stage, data._kuairand_stage_fingerprint(raw, "click", False, False))
            self.assertNotEqual(stage, data._kuairand_stage_fingerprint(raw, "all-impressions", True, False))
            self.assertNotEqual(stage, data._kuairand_stage_fingerprint(raw, "all-impressions", False, True))
            self.assertNotEqual(
                data._kuairand_filter_fingerprint(stage, 1, 5),
                data._kuairand_filter_fingerprint(stage, 2, 5),
            )
            current_fingerprint = data._kuairand_filter_fingerprint(stage, 1, 5)
            with patch.object(data, "KUAIRAND_FILTER_IMPLEMENTATION_VERSION", "future-filter-version"):
                self.assertNotEqual(current_fingerprint, data._kuairand_filter_fingerprint(stage, 1, 5))

    def test_recipe_stamps_output_name_and_metadata_after_dedup_removal(self):
        args = SimpleNamespace(
            kuairand_item_floor=555,
            kuairand_user_floor=5,
            kuairand_exclude_hate=False,
            kuairand_exclude_rand=False,
            kuairand_interaction_rule=data.KUAIRAND_DEFAULT_INTERACTION_RULE,
            kuairand_flat_output=False,
        )
        self.assertEqual(data.kuairand_output_dataset_name("kuairand-27k", args), "kuairand-27k-item555-user5")
        args.kuairand_interaction_rule = "long-view-not-hate-tab1"
        self.assertEqual(
            data.kuairand_output_dataset_name("kuairand-27k", args),
            "kuairand-27k-rule-long-view-not-hate-tab1-item555-user5",
        )
        args.kuairand_exclude_rand = True
        self.assertEqual(
            data.kuairand_output_dataset_name("kuairand-27k", args),
            "kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand",
        )
        with tempfile.TemporaryDirectory() as tmp:
            data.write_kuairand_recipe_metadata(tmp, "kuairand-27k", args)
            recipe = json.loads(Path(tmp, "recipe.json").read_text())
            self.assertEqual(recipe["item_floor"], 555)
            self.assertNotIn("dedup", recipe)
            self.assertTrue(recipe["exclude_rand"])
            self.assertIn("tab_scope_note", recipe)

    def test_interaction_rules_and_optional_rand_filter(self):
        self.assertTrue(data.kuairand_include_interaction("1"))
        self.assertFalse(data.kuairand_include_interaction("1", exclude_hate=True))
        self.assertFalse(data.kuairand_include_interaction("0", is_rand="1", exclude_rand=True))
        self.assertTrue(data.kuairand_include_interaction("0", interaction_rule="long-view-not-hate", long_view="1"))
        self.assertFalse(data.kuairand_include_interaction("1", interaction_rule="long-view-not-hate", long_view="1"))
        self.assertFalse(data.kuairand_include_interaction("0", interaction_rule="long-view-not-hate", long_view="0"))
        self.assertTrue(
            data.kuairand_include_interaction("0", interaction_rule="long-view-not-hate-tab1", long_view="1", tab="1")
        )
        self.assertFalse(
            data.kuairand_include_interaction("0", interaction_rule="long-view-not-hate-tab1", long_view="1", tab="0")
        )
        self.assertTrue(data.kuairand_include_interaction("0", interaction_rule="click", is_click="1"))
        self.assertTrue(data.kuairand_include_interaction("1", interaction_rule="click", is_click="1"))
        self.assertFalse(data.kuairand_include_interaction("0", interaction_rule="click", is_click="0"))
        self.assertTrue(data.kuairand_include_interaction("0", interaction_rule="click-not-hate", is_click="1"))
        self.assertFalse(data.kuairand_include_interaction("1", interaction_rule="click-not-hate", is_click="1"))
        self.assertTrue(data.kuairand_include_interaction("0", interaction_rule="click-tab1", is_click="1", tab="1"))
        self.assertFalse(data.kuairand_include_interaction("0", interaction_rule="click-tab1", is_click="1", tab="0"))
        self.assertTrue(
            data.kuairand_include_interaction("0", interaction_rule="click-not-hate-tab1", is_click="1", tab="1")
        )
        self.assertFalse(
            data.kuairand_include_interaction("1", interaction_rule="click-not-hate-tab1", is_click="1", tab="1")
        )
        self.assertFalse(data.kuairand_include_interaction("0", exclude_rand=True, is_rand=None))

    def test_duckdb_stage_materialization_matches_interaction_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408, "long_view": 1, "tab": 1},
                    {
                        "user_id": 1,
                        "video_id": 11,
                        "time_ms": 2,
                        "date": 20220408,
                        "long_view": 1,
                        "tab": 1,
                        "is_hate": 1,
                    },
                    {"user_id": 2, "video_id": 12, "time_ms": 3, "date": 20220408, "long_view": 1, "tab": 0},
                    {
                        "user_id": 2,
                        "video_id": 13,
                        "time_ms": 4,
                        "date": 20220408,
                        "long_view": 1,
                        "tab": 1,
                        "is_rand": 1,
                    },
                    {"user_id": 3, "video_id": 14, "time_ms": 5, "date": 20220408, "long_view": 0, "tab": 1},
                ],
            )
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                raw = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                data._ensure_kuairand_stage(
                    connection,
                    processed,
                    [str(csv_path)],
                    raw,
                    "long-view-not-hate-tab1",
                    exclude_hate=False,
                    exclude_rand=True,
                )
                staged_items = [
                    row[0] for row in connection.execute("SELECT item_id FROM staged ORDER BY item_id").fetchall()
                ]
            finally:
                connection.close()
            self.assertEqual(staged_items, [10])

    def test_duckdb_stage_materialization_matches_click_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408, "is_click": 1, "tab": 1},
                    {
                        "user_id": 1,
                        "video_id": 11,
                        "time_ms": 2,
                        "date": 20220408,
                        "is_click": 1,
                        "tab": 1,
                        "is_hate": 1,
                    },
                    {"user_id": 2, "video_id": 12, "time_ms": 3, "date": 20220408, "is_click": 1, "tab": 0},
                    {
                        "user_id": 2,
                        "video_id": 13,
                        "time_ms": 4,
                        "date": 20220408,
                        "is_click": 1,
                        "tab": 1,
                        "is_rand": 1,
                    },
                    {"user_id": 3, "video_id": 14, "time_ms": 5, "date": 20220408, "is_click": 0, "tab": 1},
                ],
            )
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                raw = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                data._ensure_kuairand_stage(
                    connection,
                    processed,
                    [str(csv_path)],
                    raw,
                    "click-not-hate-tab1",
                    exclude_hate=False,
                    exclude_rand=True,
                )
                staged_items = [
                    row[0] for row in connection.execute("SELECT item_id FROM staged ORDER BY item_id").fetchall()
                ]
            finally:
                connection.close()
            self.assertEqual(staged_items, [10])

    def test_raw_parquet_conversion_reuses_completed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408},
                    {"user_id": 1, "video_id": 11, "time_ms": 1, "date": 20220408, "is_rand": 1},
                ],
            )
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            calls = []
            old_callback = data.KUAIRAND_STAGE_FILE_CALLBACK
            try:
                data.KUAIRAND_STAGE_FILE_CALLBACK = calls.append
                raw = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                rows = connection.execute(
                    f"SELECT source_file, user_id, item_id FROM read_parquet('{data._kuairand_parquet_path(processed, csv_path.name)}') ORDER BY item_id"
                ).fetchall()
            finally:
                data.KUAIRAND_STAGE_FILE_CALLBACK = old_callback
                connection.close()
            self.assertEqual(calls, [csv_path.name])
            self.assertEqual(len(raw), 64)
            self.assertEqual(rows, [(csv_path.name, 1, 10), (csv_path.name, 1, 11)])

    def test_raw_parquet_cache_rebuilds_after_same_size_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(csv_path, [{"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408}])
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                first_fingerprint = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                first_rows = connection.execute(
                    f"SELECT item_id FROM read_parquet('{data._kuairand_parquet_path(processed, csv_path.name)}')"
                ).fetchall()

                original = csv_path.read_bytes()
                changed = original.replace(b",10,", b",11,")
                self.assertEqual(len(changed), len(original))
                csv_path.write_bytes(changed)

                second_fingerprint = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                second_rows = connection.execute(
                    f"SELECT item_id FROM read_parquet('{data._kuairand_parquet_path(processed, csv_path.name)}')"
                ).fetchall()
            finally:
                connection.close()
            self.assertNotEqual(first_fingerprint, second_fingerprint)
            self.assertEqual(first_rows, [(10,)])
            self.assertEqual(second_rows, [(11,)])

    def test_raw_parquet_equal_timestamp_order_is_stable_across_conversions(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408},
                    {"user_id": 1, "video_id": 11, "time_ms": 1, "date": 20220408},
                ],
            )
            processed_a = Path(tmp, "processed-a")
            processed_b = Path(tmp, "processed-b")
            connection_a = data._kuairand_connect(data._kuairand_duckdb_path(str(processed_a)))
            connection_b = data._kuairand_connect(data._kuairand_duckdb_path(str(processed_b)))
            try:
                data._ensure_kuairand_raw_parquet(connection_a, [str(csv_path)], "pure", str(processed_a))
                data._ensure_kuairand_raw_parquet(connection_b, [str(csv_path)], "pure", str(processed_b))
                rows_a = connection_a.execute(
                    f"SELECT source_file, item_id FROM read_parquet('{data._kuairand_parquet_path(str(processed_a), csv_path.name)}') ORDER BY item_id"
                ).fetchall()
                rows_b = connection_b.execute(
                    f"SELECT source_file, item_id FROM read_parquet('{data._kuairand_parquet_path(str(processed_b), csv_path.name)}') ORDER BY item_id"
                ).fetchall()
            finally:
                connection_a.close()
                connection_b.close()
            self.assertEqual(rows_a, [(csv_path.name, 10), (csv_path.name, 11)])
            self.assertEqual(rows_b, rows_a)

    def test_load_kuairand_orders_equal_timestamps_by_item_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 11, "time_ms": 1, "date": 20220408},
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408},
                ],
            )
            with patch.object(data, "download_kuairand", return_value=[str(csv_path)]):
                interactions = data.load_kuairand_raw(tmp, "pure", item_floor=1, user_floor=1)
            self.assertEqual(interactions, [("1", "10", 1), ("1", "11", 1)])

    def test_raw_parquet_reuse_recovers_missing_row_count_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(csv_path, [{"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408}])
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                data._kuairand_set_metadata(connection, {"parquet_file_rows": {}})
                data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                metadata = data._kuairand_metadata(connection)
            finally:
                connection.close()
            self.assertEqual(json.loads(metadata["parquet_file_rows"])[csv_path.name], 1)

    def test_raw_source_counts_reuse_per_file_parquet_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "date": 20220408, "is_hate": 1},
                    {"user_id": 2, "video_id": 20, "time_ms": 2, "date": 20220408, "is_rand": 1},
                ],
            )
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                raw = data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                connection.execute("DELETE FROM metadata WHERE key IN ('source_counts', 'source_counts_fingerprint')")
                with patch.object(data, "_kuairand_parquet_list", side_effect=AssertionError("unexpected scan")):
                    counts = data._kuairand_raw_source_counts(connection, processed, [str(csv_path)], raw)
            finally:
                connection.close()
            self.assertEqual(counts, {"hate_interactions": 1, "rand_interactions": 1})

    def test_raw_parquet_conversion_tolerates_unused_extra_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "log_standard_4_08_to_4_21_pure.csv")
            csv_path.write_text(
                ",".join([*KUAIRAND_HEADER, "unused_feature"])
                + "\n"
                + ",".join(
                    ["1", "10", "1", "20220408", "0", "1", "0", "0", "0", "0", "0", "0", "1", "10", "10", "extra"]
                )
                + "\n"
            )
            processed = data._kuairand_processed_dir(tmp, "pure")
            connection = data._kuairand_connect(data._kuairand_duckdb_path(processed))
            try:
                data._ensure_kuairand_raw_parquet(connection, [str(csv_path)], "pure", processed)
                rows = connection.execute(
                    f"SELECT source_file, user_id, item_id, tab FROM read_parquet('{data._kuairand_parquet_path(processed, csv_path.name)}')"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(rows, [(csv_path.name, 1, 10, 1)])

    def test_materialization_filtering_analysis_and_export_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "synthetic.csv")
            write_kuairand_csv(
                csv_path,
                [
                    {"user_id": 1, "video_id": 10, "time_ms": 1, "long_view": 1, "tab": 1},
                    {"user_id": 1, "video_id": 10, "time_ms": 2, "long_view": 1, "tab": 1},
                    {"user_id": 2, "video_id": 10, "time_ms": 3, "long_view": 1, "tab": 1},
                    {"user_id": 2, "video_id": 20, "time_ms": 4, "long_view": 1, "tab": 1, "is_rand": 1},
                    {"user_id": 3, "video_id": 30, "time_ms": 5, "long_view": 1, "tab": 0},
                    {"user_id": 4, "video_id": 40, "time_ms": 6, "long_view": 1, "tab": 1, "is_hate": 1},
                ],
            )
            calls = []
            cache_calls = []
            old_callback = data.KUAIRAND_STAGE_FILE_CALLBACK
            old_cache_callback = data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK
            try:
                data.KUAIRAND_STAGE_FILE_CALLBACK = calls.append
                data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK = cache_calls.append
                with patch.object(data, "download_kuairand", return_value=[str(csv_path)]):
                    result = data.load_kuairand_raw(
                        tmp,
                        variant="pure",
                        item_floor=1,
                        user_floor=1,
                        analysis_only=True,
                        interaction_rule="long-view-not-hate-tab1",
                        exclude_rand=True,
                    )
            finally:
                data.KUAIRAND_STAGE_FILE_CALLBACK = old_callback
                data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK = old_cache_callback
            self.assertIsNone(result)
            self.assertEqual(calls, ["synthetic.csv", "duckdb-stage"])
            self.assertEqual(cache_calls, ["staged_item_counts", "filtered recipe"])
            self.assertTrue(Path(tmp, "kuairand-pure", "kuairand.duckdb").exists())
            output_dir = Path(tmp, "kuairand-pure-rule-long-view-not-hate-tab1-item1-user1-exclude-rand")
            analysis = json.loads((output_dir / "analysis.json").read_text())
            self.assertEqual(analysis["stage_stats"]["interactions"], 3)
            self.assertEqual(analysis["final_stats"]["interactions"], 3)
            self.assertEqual(analysis["source_hate_interactions"], 1)
            self.assertEqual(analysis["source_rand_interactions"], 1)
            self.assertEqual(analysis["filtered_rand_distribution"], [{"value": 0, "interactions": 3}])
            self.assertFalse((output_dir / "split.json").exists())
            self.assertEqual(list(output_dir.glob("*.txt")), [])
            self.assertTrue(Path(tmp, "kuairand-pure", "standard-parquet", "synthetic.parquet").exists())
            self.assertTrue(
                Path(tmp, "kuairand-pure", "item_frequency_profile-long-view-not-hate-tab1-exclude-rand.csv").exists()
            )

    def test_missing_csv_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "bad.csv")
            csv_path.write_text("user_id,video_id,time_ms\n1,10,1\n")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                data._validate_kuairand_csv_columns(str(csv_path))

    def test_duckdb_filtering_does_not_mutate_staged_and_tie_report_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            connection = data._kuairand_connect(str(Path(tmp, "kuairand.duckdb")))
            try:
                connection.execute(
                    """
                    CREATE TABLE staged AS
                    SELECT *
                    FROM (
                        VALUES
                            ('a.csv', 1, 10, 1, 20220408, 0, 1, 0, 0, 0, 0, 0, 0, 1, 100, 100),
                            ('a.csv', 1, 10, 2, 20220408, 0, 1, 0, 0, 0, 0, 0, 0, 1, 100, 100),
                            ('a.csv', 2, 10, 3, 20220408, 0, 1, 0, 0, 0, 0, 0, 0, 1, 100, 100),
                            ('a.csv', 2, 20, 3, 20220408, 0, 1, 0, 0, 0, 0, 0, 0, 1, 100, 100),
                            ('a.csv', 3, 30, 4, 20220408, 0, 1, 0, 0, 0, 0, 0, 0, 1, 100, 100)
                    ) AS t(
                        source_file, user_id, item_id, time_ms, date,
                        is_hate, long_view, is_click, is_like, is_follow, is_comment, is_forward,
                        is_rand, tab, play_time_ms, duration_ms
                    )
                    """
                )
                data._ensure_kuairand_item_counts(connection, "stage-a")
                data._filter_kuairand_staging(connection, item_floor=2, user_floor=2)
                self.assertEqual(data._kuairand_stats(connection, "staged")["interactions"], 5)
                self.assertEqual(data._kuairand_stats(connection, "filtered")["interactions"], 2)
                stats = data._kuairand_final_timestamp_tie_stats(connection)
                self.assertEqual(stats["users"], 1)
                self.assertEqual(stats["tied_users"], 0)
            finally:
                connection.close()

    def test_27k_checksum_report_includes_deltas_and_hints(self):
        output = io.StringIO()
        with redirect_stdout(output):
            data._report_kuairand_alpha_diagnostic(
                "27k",
                {
                    "users": data.KUAIRAND_FUXI_ALPHA_STATS["users"] + 1,
                    "items": data.KUAIRAND_FUXI_ALPHA_STATS["items"] + 1,
                    "interactions": data.KUAIRAND_FUXI_ALPHA_STATS["interactions"] - 1,
                    "avg_length": data.KUAIRAND_FUXI_ALPHA_STATS["avg_length"] - 1,
                },
            )
        text = output.getvalue()
        self.assertIn("Delta users", text)
        self.assertIn("Delta items", text)
        self.assertIn("raise --kuairand-item-floor", text)
        self.assertIn("raise --kuairand-user-floor", text)

    def test_reduced_rows_use_existing_split_contract(self):
        interactions = [
            ("1", "10", 1),
            ("1", "11", 2),
            ("1", "12", 2),
            ("2", "10", 1),
            ("2", "11", 2),
            ("2", "12", 3),
        ]
        split = data.reindex_and_split(interactions)
        self.assertEqual(split.num_users, 2)
        self.assertEqual(split.num_items, 3)
        self.assertEqual(split.user_train[1], [1])
        self.assertEqual(split.user_valid[1], [2])
        self.assertEqual(split.user_test[1], [3])
        self.assertEqual(split.user_test_ts[1], [2])


if __name__ == "__main__":
    unittest.main()
