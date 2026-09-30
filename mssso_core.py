"""Core data and optimization routines for the MSSSO demonstration."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


INTEL_COLUMNS = [
    "date",
    "time",
    "epoch",
    "sensor_id",
    "temperature",
    "humidity",
    "light",
    "voltage",
]
METADATA_COLUMNS = {"sensor_id", "timestamp", "date", "time", "epoch", "x", "y"}


@dataclass
class SensorCube:
    sensor_ids: list[str]
    timestamps: pd.DatetimeIndex
    variables: list[str]
    values: np.ndarray
    coordinates: np.ndarray
    train_end: int
    synthetic_coordinates: bool


def _read_payload(source: object) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        payload = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        return payload
    raise ValueError("Unsupported file input.")


def _intel_frame(payload: bytes, compressed: bool) -> pd.DataFrame:
    source = io.BytesIO(payload) if compressed else io.StringIO(payload.decode("utf-8", errors="replace"))
    chunks = pd.read_csv(
        source,
        compression="gzip" if compressed else None,
        sep=r"\s+",
        header=None,
        names=INTEL_COLUMNS,
        usecols=[0, 1, 3, 4, 5, 6, 7],
        on_bad_lines="skip",
        chunksize=200_000,
    )
    measurement_names = ["temperature", "humidity", "light", "voltage"]
    partial_sums: list[pd.DataFrame] = []
    partial_counts: list[pd.DataFrame] = []

    for chunk in chunks:
        chunk["timestamp"] = pd.to_datetime(
            chunk["date"].astype(str) + " " + chunk["time"].astype(str), errors="coerce"
        ).dt.floor("30min")
        numeric_ids = pd.to_numeric(chunk["sensor_id"], errors="coerce")
        valid_ids = numeric_ids.notna() & numeric_ids.mod(1).eq(0)
        chunk = chunk.loc[valid_ids].copy()
        chunk["sensor_id"] = numeric_ids.loc[valid_ids].astype("int64").astype(str)
        chunk = chunk.dropna(subset=["timestamp"])
        for name in measurement_names:
            chunk[name] = pd.to_numeric(chunk[name], errors="coerce")
        grouped = chunk.groupby(["sensor_id", "timestamp"])[measurement_names]
        partial_sums.append(grouped.sum(min_count=1))
        partial_counts.append(grouped.count())

    sums = pd.concat(partial_sums).groupby(level=[0, 1]).sum(min_count=1)
    counts = pd.concat(partial_counts).groupby(level=[0, 1]).sum()
    return sums.div(counts.where(counts > 0)).reset_index()


def _normalize_column_name(value: object) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def _read_locations(source: object) -> pd.DataFrame:
    payload = _read_payload(source)
    try:
        locations = pd.read_csv(io.BytesIO(payload))
        normalized = {_normalize_column_name(column): column for column in locations.columns}
        id_column = next(
            (normalized[name] for name in ("sensorid", "moteid", "nodeid", "id") if name in normalized),
            None,
        )
        x_column = normalized.get("x")
        y_column = normalized.get("y")
        if id_column is not None and x_column is not None and y_column is not None:
            return locations.rename(columns={id_column: "sensor_id", x_column: "x", y_column: "y"})[
                ["sensor_id", "x", "y"]
            ]
    except (UnicodeDecodeError, pd.errors.ParserError):
        pass

    locations = pd.read_csv(
        io.StringIO(payload.decode("utf-8", errors="replace")),
        sep=r"\s+",
        header=None,
        names=["sensor_id", "x", "y"],
        usecols=[0, 1, 2],
    )
    return locations


def load_sensor_data(
    source: object,
    filename: str = "readings.csv",
    locations_source: object | None = None,
) -> pd.DataFrame:
    """Load a wide sensor CSV or the official Intel Lab whitespace file."""
    payload = _read_payload(source)
    lower_name = filename.lower()
    is_intel = lower_name.endswith((".txt", ".txt.gz", ".gz"))

    if is_intel:
        frame = _intel_frame(payload, compressed=lower_name.endswith(".gz"))
    else:
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except UnicodeDecodeError as error:
            raise ValueError("Could not read this file as CSV. Use CSV or the Intel .txt.gz format.") from error
        frame = frame.rename(columns={column: str(column).strip() for column in frame.columns})
        normalized = {_normalize_column_name(column): column for column in frame.columns}
        aliases = {
            "sensorid": "sensor_id",
            "moteid": "sensor_id",
            "nodeid": "sensor_id",
            "datetime": "timestamp",
            "datetimestamp": "timestamp",
            "timeindex": "timestamp",
        }
        rename = {column: aliases[key] for key, column in normalized.items() if key in aliases}
        frame = frame.rename(columns=rename)
        normalized = {_normalize_column_name(column): column for column in frame.columns}
        if "timestamp" not in frame.columns and "date" in normalized and "time" in normalized:
            date_column, time_column = normalized["date"], normalized["time"]
            frame["timestamp"] = frame[date_column].astype(str) + " " + frame[time_column].astype(str)

    if "sensor_id" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("The readings file needs sensor_id and timestamp columns (or Intel Lab format).")

    frame["sensor_id"] = frame["sensor_id"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["sensor_id", "timestamp"])

    if locations_source is not None:
        locations = _read_locations(locations_source)
        locations["sensor_id"] = locations["sensor_id"].astype(str).str.strip()
        frame = frame.drop(columns=[name for name in ("x", "y") if name in frame.columns])
        frame = frame.merge(locations, on="sensor_id", how="left")
        if is_intel:
            frame = frame.dropna(subset=["x", "y"])

    excluded = METADATA_COLUMNS | {"latitude", "longitude", "lat", "lon", "moteid"}
    for column in frame.columns:
        if column not in excluded:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    measurement_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()
    ]
    if not measurement_columns:
        raise ValueError("No numeric measurement columns were found in the readings file.")

    frame = frame.dropna(subset=measurement_columns, how="all")
    if is_intel:
        frame["timestamp"] = frame["timestamp"].dt.floor("30min")
        frame = frame.groupby(["sensor_id", "timestamp"], as_index=False)[measurement_columns + [
            column for column in ("x", "y") if column in frame.columns
        ]].mean(numeric_only=True)
    return frame


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS | {"latitude", "longitude", "lat", "lon"}
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]


def make_demo_data(sensor_count: int = 16, time_count: int = 84) -> pd.DataFrame:
    """Return a deterministic synthetic multivariate environmental network."""
    random = np.random.default_rng(2026)
    columns = 4
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2026-04-01", periods=time_count, freq="h")
    phase = np.arange(time_count)

    for sensor in range(sensor_count):
        x = float(sensor % columns)
        y = float(sensor // columns)
        local_phase = (x * 0.23) - (y * 0.17)
        for step, timestamp in enumerate(timestamps):
            daily = 2 * math.pi * step / 24
            plume = math.exp(-((x - 1.2) ** 2 + (y - 1.7) ** 2) / 1.8)
            rows.append(
                {
                    "sensor_id": f"S{sensor + 1:02d}",
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                    "temperature": 21.5 + 3.2 * math.sin(daily + local_phase) + 0.35 * y
                    + random.normal(0, 0.24),
                    "humidity": 57 - 8.5 * math.sin(daily + local_phase + 0.55)
                    + 1.1 * math.cos(step / 11 + x) + random.normal(0, 0.8),
                    "light": max(
                        8,
                        260 + 150 * math.sin(daily - 1.15 + local_phase) + 95 * plume * math.sin(step / 7)
                        + random.normal(0, 12),
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_sensor_cube(
    frame: pd.DataFrame,
    variables: list[str],
    train_fraction: float = 0.7,
    max_timestamps: int = 180,
) -> SensorCube:
    if not variables:
        raise ValueError("Choose at least one environmental variable.")
    if "sensor_id" not in frame or "timestamp" not in frame:
        raise ValueError("The dataset must have sensor_id and timestamp columns.")

    sensor_ids = sorted(frame["sensor_id"].astype(str).unique().tolist())
    timestamps = pd.DatetimeIndex(sorted(pd.to_datetime(frame["timestamp"]).dropna().unique()))
    if len(sensor_ids) < 3 or len(timestamps) < 10:
        raise ValueError("At least 3 sensors and 10 timestamps are needed for this demonstration.")
    if len(timestamps) > max_timestamps:
        sample_indices = np.linspace(0, len(timestamps) - 1, max_timestamps, dtype=int)
        timestamps = timestamps[sample_indices]
        frame = frame[frame["timestamp"].isin(timestamps)]

    values_by_variable: list[np.ndarray] = []
    for variable in variables:
        if variable not in frame.columns:
            raise ValueError(f"Measurement column '{variable}' is missing.")
        wide = frame.pivot_table(
            index="timestamp", columns="sensor_id", values=variable, aggfunc="mean"
        ).reindex(index=timestamps, columns=sensor_ids)
        wide = wide.interpolate(axis=0, limit_direction="both")
        wide = wide.fillna(wide.median(axis=0)).fillna(0.0)
        values_by_variable.append(wide.to_numpy(dtype=float).T)

    values = np.stack(values_by_variable, axis=2)
    synthetic_coordinates = not {"x", "y"}.issubset(frame.columns)
    if not synthetic_coordinates:
        coordinates_frame = frame.groupby("sensor_id")[["x", "y"]].median().reindex(sensor_ids)
        if coordinates_frame.isna().any().any():
            synthetic_coordinates = True
    if synthetic_coordinates:
        width = max(1, math.ceil(math.sqrt(len(sensor_ids))))
        coordinates = np.array([[index % width, index // width] for index in range(len(sensor_ids))], dtype=float)
    else:
        coordinates = coordinates_frame.to_numpy(dtype=float)

    train_end = min(len(timestamps) - 1, max(3, int(round(len(timestamps) * train_fraction))))
    return SensorCube(sensor_ids, timestamps, list(variables), values, coordinates, train_end, synthetic_coordinates)


def _dtw_distance(first: np.ndarray, second: np.ndarray, window: int | None = None) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    rows, columns = len(first), len(second)
    if rows == 0 or columns == 0:
        return 0.0
    band = max(abs(rows - columns), window if window is not None else max(rows, columns))
    previous = np.full(columns + 1, np.inf)
    previous[0] = 0.0
    for row in range(1, rows + 1):
        current = np.full(columns + 1, np.inf)
        start = max(1, row - band)
        end = min(columns, row + band)
        for column in range(start, end + 1):
            cost = abs(first[row - 1] - second[column - 1])
            current[column] = cost + min(previous[column], current[column - 1], previous[column - 1])
        previous = current
    return float(previous[columns] / (rows + columns))


def compute_dtw_matrices(values: np.ndarray, max_series_points: int = 96) -> list[np.ndarray]:
    """Compute variable-wise pairwise DTW distances using training timestamps only."""
    sensor_count, time_count, variable_count = values.shape
    sample_indices = np.linspace(0, time_count - 1, min(time_count, max_series_points), dtype=int)
    matrices: list[np.ndarray] = []
    for variable in range(variable_count):
        series = values[:, sample_indices, variable].astype(float)
        means = series.mean(axis=1, keepdims=True)
        scales = series.std(axis=1, keepdims=True)
        normalized = np.divide(series - means, scales, out=np.zeros_like(series), where=scales > 1e-12)
        distances = np.zeros((sensor_count, sensor_count), dtype=float)
        band = max(2, normalized.shape[1] // 8)
        for first in range(sensor_count):
            for second in range(first + 1, sensor_count):
                distance = _dtw_distance(normalized[first], normalized[second], window=band)
                distances[first, second] = distance
                distances[second, first] = distance
        maximum = float(distances.max())
        if maximum > 0:
            distances /= maximum
        matrices.append(distances)
    return matrices


def tsem_entropy(subset: tuple[int, ...] | list[int], distances: list[np.ndarray], bins: int = 10) -> float:
    """Mean normalized entropy of selected-sensor DTW distance distributions."""
    indices = np.asarray(sorted(set(subset)), dtype=int)
    if len(indices) < 2:
        return 0.0
    pair_rows, pair_columns = np.triu_indices(len(indices), k=1)
    scores = []
    for matrix in distances:
        pair_distances = matrix[np.ix_(indices, indices)][pair_rows, pair_columns]
        counts, _ = np.histogram(pair_distances, bins=bins, range=(0.0, 1.0))
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        scores.append(entropy / math.log(bins))
    return float(np.mean(scores)) if scores else 0.0


def optimize_ga(
    distances: list[np.ndarray],
    subset_size: int,
    population_size: int = 36,
    generations: int = 24,
    mutation_rate: float = 0.25,
    seed: int = 42,
) -> tuple[list[int], list[float], float]:
    sensor_count = distances[0].shape[0]
    if not 1 <= subset_size <= sensor_count:
        raise ValueError("Subset size must be between 1 and the total number of sensors.")
    if subset_size == sensor_count:
        subset = tuple(range(sensor_count))
        score = tsem_entropy(subset, distances)
        return list(subset), [score] * (generations + 1), score

    random = np.random.default_rng(seed)
    population_size = max(4, population_size)
    population: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        candidate = tuple(sorted(random.choice(sensor_count, size=subset_size, replace=False).tolist()))
        if candidate not in seen:
            seen.add(candidate)
            population.append(candidate)

    cache: dict[tuple[int, ...], float] = {}

    def fitness(candidate: tuple[int, ...]) -> float:
        if candidate not in cache:
            cache[candidate] = tsem_entropy(candidate, distances)
        return cache[candidate]

    best = max(population, key=fitness)
    best_score = fitness(best)
    history = [best_score]

    def parent() -> tuple[int, ...]:
        contestants = random.choice(population, size=min(3, len(population)), replace=False)
        return max((tuple(candidate) for candidate in contestants), key=fitness)

    for _ in range(generations):
        next_population = [best]
        while len(next_population) < population_size:
            first, second = parent(), parent()
            pool = list(set(first) | set(second))
            if len(pool) < subset_size:
                pool.extend(index for index in range(sensor_count) if index not in pool)
            child = set(random.choice(pool, size=subset_size, replace=False).tolist())
            if random.random() < mutation_rate:
                available = [index for index in range(sensor_count) if index not in child]
                if available:
                    child.remove(int(random.choice(tuple(child))))
                    child.add(int(random.choice(available)))
            next_population.append(tuple(sorted(child)))
        population = next_population
        generation_best = max(population, key=fitness)
        generation_score = fitness(generation_best)
        if generation_score > best_score:
            best, best_score = generation_best, generation_score
        history.append(best_score)

    return list(best), history, float(best_score)


def spatial_coverage_score(coordinates: np.ndarray, selected: list[int]) -> float:
    """Return 1 - normalized mean nearest-selected distance (a prototype proxy)."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(selected) == len(coordinates):
        return 1.0
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[np.asarray(selected)][None, :, :], axis=2)
    diagonal = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    if diagonal <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - distances.min(axis=1).mean() / diagonal, 0.0, 1.0))


def evaluate_gpr(cube: SensorCube, selected: list[int]) -> dict[str, object]:
    """Reconstruct unselected sensors from selected sensors during the test window."""
    holdout = [index for index in range(len(cube.sensor_ids)) if index not in selected]
    if not holdout:
        return {"overall_rmse": 0.0, "variable_rmse": {}, "actual": [], "predicted": []}

    coordinates = cube.coordinates.astype(float)
    span = np.ptp(coordinates, axis=0)
    span[span == 0] = 1.0
    coordinates = (coordinates - coordinates.min(axis=0)) / span
    predictions = np.empty((len(holdout), len(cube.timestamps) - cube.train_end, len(cube.variables)))
    actual = cube.values[holdout, cube.train_end :, :]
    kernel = RBF(length_scale=0.35, length_scale_bounds="fixed") + WhiteKernel(
        noise_level=0.04, noise_level_bounds="fixed"
    )

    for variable_index in range(len(cube.variables)):
        for test_step, time_index in enumerate(range(cube.train_end, len(cube.timestamps))):
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None)
            targets = cube.values[np.asarray(selected), time_index, variable_index]
            model.fit(coordinates[np.asarray(selected)], targets)
            predictions[:, test_step, variable_index] = model.predict(coordinates[holdout])

    variable_rmse = {
        variable: float(np.sqrt(np.mean((predictions[:, :, index] - actual[:, :, index]) ** 2)))
        for index, variable in enumerate(cube.variables)
    }
    train_scale = cube.values[:, : cube.train_end, :].std(axis=(0, 1))
    train_scale[train_scale < 1e-12] = 1.0
    standardized_error = (predictions - actual) / train_scale.reshape(1, 1, -1)
    return {
        "overall_rmse": float(np.sqrt(np.mean(standardized_error**2))),
        "variable_rmse": variable_rmse,
        "actual": actual.reshape(-1).tolist(),
        "predicted": predictions.reshape(-1).tolist(),
    }