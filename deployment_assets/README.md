# Cloud demonstration assets

`cicids2017_cloud_demo_sample.csv` is a small, class-balanced demonstration subset drawn only
from the existing processed CICIDS2017 dataset. It contains 25 records from each canonical class
(100 records total), the ordered 65 model features and the `target` label.

The subset is selected with seed 42 by assigning deterministic random keys while reading the
processed dataset in source order and retaining the 25 smallest keys per class. Selection does not
use model predictions, confidence, correctness or SHAP results. It is not an independent dataset
and no performance metric should be calculated or advertised from it.

The accompanying manifest records the method and distribution without storing local paths or
personal information.

`rf_explainability_summary.json` contains only the two frozen statistics used by the interface.
It avoids publishing the unrelated local Windows paths contained in the full experimental metadata;
the original metadata and all explanation values remain unchanged.

## Selecting the application mode

- Default: `LOCAL`, as declared in `config.yaml`. This uses the complete processed CICIDS2017 CSV.
- Cloud: set environment variable `STREAMLIT_DEPLOYMENT_MODE=CLOUD_DEMO`. This uses only the small
  file in this directory and does not access the complete processed CSV.

Both modes use the same frozen RF model, MLP model, scaler, curated explanation artefacts, original
comparison evidence and external-validation evidence.
