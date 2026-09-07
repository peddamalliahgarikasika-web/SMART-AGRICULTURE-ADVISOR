"""
generate_disease_dataset.py
-----------------------------------------------------
Builds a synthetic, labeled symptom dataset from the disease knowledge
base in disease_data.py, so a real machine-learning classifier
(RandomForest) can be trained WITHOUT needing a leaf-image dataset.

Why synthetic data instead of leaf photos?
  The CNN image classifier (train_disease_model.py) needs thousands of
  real leaf photos per class, which most users don't have lying around.
  This script instead treats every symptom combination already encoded
  in disease_data.py (crop, leaf colour, spot colour/size, leaf curl,
  powder, stem condition, temperature range, humidity range) as a
  "rule", and samples many slightly-noisy variations of each rule to
  produce a proper training table. This lets scikit-learn learn the
  same relationships the old if/else scoring engine used, but as a
  genuine trained model with probability estimates — usable instantly,
  with zero extra downloads.

Run:
    python generate_disease_dataset.py

Output:
    ml/disease_symptom_dataset.csv
"""

import os
import sys
import random
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from disease_data import disease_database  # noqa: E402

random.seed(42)

OUT_PATH = os.path.join(HERE, "disease_symptom_dataset.csv")

# How many synthetic rows to generate per disease_database entry.
ROWS_PER_ENTRY = 400

# Categorical fields that get "noisy" (occasionally swapped to a
# different valid option) so the model doesn't just memorise exact
# matches and instead learns which symptoms matter most.
LEAF_COLORS = ["Green", "Yellow", "Brown", "Black", "White"]
SPOT_COLORS = ["None", "Brown", "Black", "White", "Yellow"]
SPOT_SIZES = ["None", "Small", "Medium", "Large"]
YES_NO = ["No", "Yes"]
STEMS = ["Healthy", "Rotten", "Dry", "Cracked"]

FIELDS = [
    "crop", "leaf_color", "spot_color", "spot_size", "leaf_curl",
    "powder", "stem", "temperature", "humidity", "disease",
]


def noisy_choice(correct_value, options, correct_probability=0.82):
    """Return `correct_value` most of the time; otherwise a random
    alternative from `options`, so the dataset isn't perfectly clean
    (mirrors real farmer-entered questionnaire data)."""
    if random.random() < correct_probability:
        return correct_value
    return random.choice(options)


def main():
    rows = []

    for entry in disease_database:
        t_lo, t_hi = entry["temperature"]
        h_lo, h_hi = entry["humidity"]

        # Pad the numeric ranges a little so values near the boundary
        # are also represented, then let noise occasionally push a
        # sample outside the "ideal" range entirely.
        t_pad = max(1.0, (t_hi - t_lo) * 0.15)
        h_pad = max(1.0, (h_hi - h_lo) * 0.15)

        for _ in range(ROWS_PER_ENTRY):
            temperature = round(random.uniform(t_lo - t_pad, t_hi + t_pad), 1)
            humidity = round(random.uniform(h_lo - h_pad, h_hi + h_pad), 1)

            # Occasionally sample a wildly different weather reading so
            # the model doesn't over-index purely on weather.
            if random.random() < 0.06:
                temperature = round(random.uniform(10, 42), 1)
            if random.random() < 0.06:
                humidity = round(random.uniform(20, 100), 1)

            row = {
                "crop": entry["crop"],
                "leaf_color": noisy_choice(entry["leaf_color"], LEAF_COLORS),
                "spot_color": noisy_choice(entry["spot_color"], SPOT_COLORS),
                "spot_size": noisy_choice(entry["spot_size"], SPOT_SIZES),
                "leaf_curl": noisy_choice(entry["leaf_curl"], YES_NO, 0.88),
                "powder": noisy_choice(entry["powder"], YES_NO, 0.88),
                "stem": noisy_choice(entry["stem"], STEMS, 0.85),
                "temperature": temperature,
                "humidity": humidity,
                "disease": entry["disease"],
            }
            rows.append(row)

    os.makedirs(HERE, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic rows across "
          f"{len(disease_database)} disease entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
