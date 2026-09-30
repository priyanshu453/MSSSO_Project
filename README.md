# MSSSO Sensor Selection Lab

A runnable first-stage implementation of the project presentation's **TSEM + Genetic Algorithm** workflow. It loads multivariate sensor readings, computes per-variable pairwise Dynamic Time Warping distances, maximizes their entropy with a fixed-size GA subset, and evaluates spatial reconstruction on unselected sensors.

## Run in VS Code (Windows)

Open this folder in VS Code, then run these commands in the integrated PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Choose the `.venv` interpreter in VS Code if you want editor diagnostics to use the same environment.

## Dataset used

The project bundles the **Intel Berkeley Research Lab sensor dataset** as its default input. It contains readings from 54 Mica2Dot sensor motes collected in 2004, at roughly 31-second intervals: temperature, humidity, light, and voltage, with date/time and mote ID. The original readings are in `data/intel_lab_data.txt.gz`; sensor coordinates are in `data/mote_locs.txt`.

Official source:

- Dataset description: https://db.csail.mit.edu/labdata/labdata.html
- Readings: https://db.csail.mit.edu/labdata/data.txt.gz
- Sensor coordinates: https://db.csail.mit.edu/labdata/mote_locs.txt

The app loads this bundled dataset automatically. It groups measurements into 30-minute intervals and merges the coordinate file for the spatial view. The sidebar lets you select the four environmental measurements; date/time and mote ID are shown as record metadata and are not treated as environmental measurements.

### Use another dataset

1. Put your CSV in this project's `data/` folder, or leave it elsewhere and select it in the app's **Sensor readings** uploader.
2. For a normal CSV, use one row per sensor and timestamp. Required fields are `sensor_id`, `timestamp`, and one or more numeric measurement columns; `x` and `y` are optional.
3. For another Intel download, upload `data.txt.gz` under **Sensor readings**, then upload `mote_locs.txt` under **Sensor coordinates**.
4. Choose the measurement variables and sensor budget in the sidebar, then select **Run TSEM + GA**.

Intel's raw file is large, so the loader parses it in chunks and aggregates it into 30-minute readings. Datasets with more than 180 distinct timestamps are evenly sampled for the interactive experiment. The GA uses the first 70% of time points; GPR estimates unselected locations during the final 30% using contemporaneous readings from selected sensors. This is spatial reconstruction, not future forecasting.

## What this milestone implements

- Preprocessing of wide multivariate sensor CSVs and Intel Lab `.txt.gz` readings, optional coordinate merge, duplicate aggregation, and missing-value interpolation.
- Variable-wise DTW distance matrices, normalized Shannon-entropy TSEM fitness, and a fixed-cardinality GA with elitism, tournament selection, crossover, mutation, and best-so-far tracking.
- A seeded random-subset baseline, Gaussian-process spatial reconstruction, per-variable RMSE, pooled standardized RMSE, and a nearest-selected-sensor coverage proxy.
- Interactive sensor layout, convergence, and error visualizations.

The coverage value is labeled a **proxy** in the dashboard; the five-method benchmark, PSO, exhaustive search, and the presentation's adaptive reselection novelty are not part of this 50% milestone. The TSEM fitness follows the presentation's DTW-distance entropy description; it is not a claim of reproducing every experiment or result in the paper.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```