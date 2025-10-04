# ExoplanetHunter-TransitML

Detect exoplanet transits using classical machine learning on Kepler lightcurve data. The project leverages [Lightkurve](https://docs.lightkurve.org/) for lightcurve retrieval and preprocessing, and [grouped_timeserie_cv](https://github.com/zenta-ab/grouped_timeserie_cv) for grouped time series cross-validation.

## Installation

```bash
pip install lightkurve
pip install grouped_timeserie_cv
```

## Features

* Download and normalize Kepler/TESS lightcurves.
* Segment long lightcurves into fixed-length windows.
* Compute statistical features (mean, std, min, max, skew, kurtosis, transit depth) per segment.
* Label segments as containing a transit (`1`) or not (`0`).
* Use grouped cross-validation to avoid data leakage across targets or segments.

## Usage Example

```python
import lightkurve as lk
from transit_dataset import build_segmented_dataset

# Download and preprocess a single lightcurve
lc = lk.search_lightcurve("Kepler-10").download().remove_nans().normalize()

# Build segmented dataset
df_segments = build_segmented_dataset(lc, segment_length=200, sigma=5)
df_segments["target_id"] = "Kepler-10"

# Save to CSV
df_segments.to_csv("transit_segments.csv", index=False)
```

## Links

* [Lightkurve Documentation](https://docs.lightkurve.org/)
* [grouped_timeserie_cv GitHub](https://github.com/zenta-ab/grouped_timeserie_cv)

---
