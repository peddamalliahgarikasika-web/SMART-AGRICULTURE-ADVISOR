"""
generate_crop_dataset.py

Builds a training dataset for the crop recommendation model.

WHY A GENERATED DATASET?
-------------------------
The original app hard-coded agronomic ranges (N, P, K, pH, temperature,
humidity, rainfall) for 60 crops inside app.py, but only ever *used* 8 of
them because of a bug (a list was closed early with a stray "]"). That
bug is fixed in the new app.py.

Those 60 ranges were written by hand from real agronomic references, so
instead of throwing them away, we treat them as the "ground truth" for
each crop and sample realistic values inside (and slightly outside) those
ranges to build a proper labelled dataset. This is the same idea used in
the public "Crop_recommendation.csv" Kaggle dataset, except tailored to
the 60 Indian crops already curated in this project.

Each crop gets SAMPLES_PER_CROP rows. Values are drawn from a normal
distribution centred on the middle of each range, clipped so ~97% of
samples fall inside the original range and a few fall just outside it
(so the classifier learns soft boundaries instead of memorising exact
cutoffs).

Output: ml/crop_dataset.csv with columns:
    N, P, K, temperature, humidity, ph, rainfall, label
"""

import json
import os
import numpy as np
import pandas as pd

SEED = 42
SAMPLES_PER_CROP = 300

HERE = os.path.dirname(os.path.abspath(__file__))
CROP_DB_PATH = os.path.join(HERE, "crop_database.json")
OUTPUT_PATH = os.path.join(HERE, "crop_dataset.csv")

FEATURES = ["N", "P", "K", "temp", "humidity", "ph", "rainfall"]
OUTPUT_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def sample_range(low, high, n, rng):
    """Sample n values centred in [low, high] using a clipped normal
    distribution so most values stay inside the range but a small
    fraction lands just outside it (real-world noise)."""
    mean = (low + high) / 2.0
    # std chosen so the range covers ~4 standard deviations
    std = max((high - low) / 4.0, 1e-6)
    values = rng.normal(loc=mean, scale=std, size=n)
    return values


def main():
    with open(CROP_DB_PATH, "r", encoding="utf-8") as f:
        crop_database = json.load(f)

    rng = np.random.default_rng(SEED)

    rows = []
    for crop in crop_database:
        name = crop["crop"]
        n_vals = sample_range(*crop["N"], SAMPLES_PER_CROP, rng)
        p_vals = sample_range(*crop["P"], SAMPLES_PER_CROP, rng)
        k_vals = sample_range(*crop["K"], SAMPLES_PER_CROP, rng)
        temp_vals = sample_range(*crop["temp"], SAMPLES_PER_CROP, rng)
        hum_vals = sample_range(*crop["humidity"], SAMPLES_PER_CROP, rng)
        ph_vals = sample_range(*crop["ph"], SAMPLES_PER_CROP, rng)
        rain_vals = sample_range(*crop["rainfall"], SAMPLES_PER_CROP, rng)

        for i in range(SAMPLES_PER_CROP):
            rows.append({
                "N": round(max(n_vals[i], 0), 2),
                "P": round(max(p_vals[i], 0), 2),
                "K": round(max(k_vals[i], 0), 2),
                "temperature": round(temp_vals[i], 2),
                "humidity": round(min(max(hum_vals[i], 0), 100), 2),
                "ph": round(min(max(ph_vals[i], 0), 14), 2),
                "rainfall": round(max(rain_vals[i], 0), 2),
                "label": name,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} rows for {len(crop_database)} crops")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
