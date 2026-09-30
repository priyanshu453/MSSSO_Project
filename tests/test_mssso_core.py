import gzip
import unittest

import numpy as np

from mssso_core import (
    build_sensor_cube,
    compute_dtw_matrices,
    load_sensor_data,
    make_demo_data,
    optimize_ga,
    tsem_entropy,
    _dtw_distance,
)


class MSSSOCoreTests(unittest.TestCase):
    def test_intel_lab_gzip_and_whitespace_locations_load(self):
        readings = "\n".join(
            [
                "2004-02-28 00:00:00.000 1 1 20.0 50.0 300.0 2.7",
                "2004-02-28 00:00:00.000 1 2 21.0 51.0 310.0 2.7",
                "2004-02-28 00:00:00.000 1 3 22.0 52.0 320.0 2.7",
                "2004-02-28 00:31:00.000 2 1 20.5 50.5 305.0 2.7",
                "2004-02-28 00:31:00.000 2 2 21.5 51.5 315.0 2.7",
                "2004-02-28 00:31:00.000 2 3 22.5 52.5 325.0 2.7",
                "2004-02-28 00:00:00.000 1 51.0 23.0 53.0 330.0 2.7",
                "2004-02-28 00:31:00.000 2 51.0 23.5 53.5 335.0 2.7",
                "2004-02-28 00:00:00.000 1 55 24.0 54.0 340.0 2.7",
            ]
        ).encode()
        locations = b"1 0.0 0.0\n2 2.0 0.0\n3 1.0 1.0\n51 3.0 1.0\n"

        frame = load_sensor_data(gzip.compress(readings), "intel.txt.gz", locations)

        self.assertEqual(set(frame["sensor_id"]), {"1", "2", "3", "51"})
        self.assertEqual(len(frame), 8)
        self.assertTrue({"temperature", "humidity", "light", "voltage", "x", "y"}.issubset(frame.columns))
        self.assertEqual(frame["timestamp"].nunique(), 2)

    def test_demo_data_builds_multivariate_cube(self):
        frame = make_demo_data(sensor_count=8, time_count=24)
        cube = build_sensor_cube(frame, ["temperature", "humidity"])

        self.assertEqual(cube.values.shape, (8, 24, 2))
        self.assertEqual(cube.train_end, 17)
        self.assertFalse(np.isnan(cube.values).any())
        self.assertFalse(cube.synthetic_coordinates)

    def test_dtw_is_zero_for_identical_sequences(self):
        sequence = np.array([0.0, 0.5, -0.2, 0.8])
        self.assertEqual(_dtw_distance(sequence, sequence), 0.0)

    def test_ga_returns_fixed_unique_subset_and_monotonic_history(self):
        frame = make_demo_data(sensor_count=9, time_count=28)
        cube = build_sensor_cube(frame, ["temperature", "humidity"])
        matrices = compute_dtw_matrices(cube.values[:, : cube.train_end, :], max_series_points=24)
        selected, history, score = optimize_ga(matrices, 4, population_size=12, generations=8, seed=9)

        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(all(0 <= index < 9 for index in selected))
        self.assertTrue(all(later >= earlier for earlier, later in zip(history, history[1:])))
        self.assertAlmostEqual(score, tsem_entropy(selected, matrices))


if __name__ == "__main__":
    unittest.main()