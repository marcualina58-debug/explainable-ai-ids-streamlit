# Streamlit Community Cloud staging package

This directory contains only the runtime files required for the seven-tab `CLOUD_DEMO` application.
It has not been published or deployed.

## Entry point

`app/app.py`

## Python and dependencies

Use Python **3.13**. The frozen models and complete staged application were validated locally with
Python 3.13.15 and the exact package versions in `requirements.txt`. TensorFlow 2.21.0 declares
Python 3.10 or newer, but the eventual Streamlit Community Cloud environment must still be tested.

## Deployment mode

`config.yaml` defaults to `CLOUD_DEMO` in this staging package. The original project remains in
`LOCAL` mode. The custom-record demonstration reads only
`deployment_assets/cicids2017_cloud_demo_sample.csv`; no complete CICIDS2017 or CSE-CIC-IDS2018
dataset is included.

The 100-record sample is a balanced demonstration subset, not an independent evaluation dataset.
It must not be used to calculate or advertise performance metrics. The provider's dataset FAQ
permits redistribution when the dataset and associated research paper are cited. The required
attribution is retained in `DATASET_ATTRIBUTION.md` and must accompany any public package.

## Local validation command

From this directory, using an existing compatible environment:

```powershell
python -m streamlit run app/app.py
```

The package contains the original frozen models and only the saved results read by the interface.
It contains no credentials, raw datasets, full processed dataset, virtual environment, notebook,
test cache or personal execution log.
