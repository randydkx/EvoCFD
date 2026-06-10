# EvoCFD: Continual Fraud Detection with Dynamic Feature Space

This repository contains the official code for **"Continual Fraud Detection with Dynamic Feature Space"**. The repository is prepared for the ICDE 2027 review process.

EvoCFD targets continual fraud detection when the available feature space changes over time. The included IEEE workflow uses the public IEEE-CIS Fraud Detection data, cleans it into the format expected by the training code, then trains EvoCFD with the stage definitions stored under `assets/stage_exports`.

## Repository Layout

- `run_experiment.py`: main EvoCFD training and evaluation entry point.
- `run_ieee.sh`: recommended script for reproducing the IEEE experiment.
- `clean_data.ipynb`: notebook that converts the raw IEEE-CIS files into the cleaned CSV consumed by `run_experiment.py`.
- `assets/stage_exports/ieee_stage_summary.json`: fixed IEEE stage metadata, including stage windows and feature sets.
- `assets/stage_exports/ieee_feature_processing_audit.json`: audit snapshot for the cleaned IEEE feature processing.
- `model/dlmodel/model/methods/EvoCFD.py`: EvoCFD method implementation.
- `model/dlmodel/model/models/FTT.py`: tabular Transformer backbone used by EvoCFD.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

**The release of the proprietary PerFraud dataset and the MerFraud dataset are under check.** The experiment is based on the Kaggle IEEE-CIS Fraud Detection dataset (publicly available). Download it from:

```bash
export EVO_CFD_DATA_ROOT=/data
mkdir -p ${EVO_CFD_DATA_ROOT}/ieee-fraud-detection
kaggle competitions download -c ieee-fraud-detection -p ${EVO_CFD_DATA_ROOT}
unzip ${EVO_CFD_DATA_ROOT}/ieee-fraud-detection.zip \
  -d ${EVO_CFD_DATA_ROOT}/ieee-fraud-detection
```

After extraction, the directory should contain at least:

```text
${EVO_CFD_DATA_ROOT}/ieee-fraud-detection/train_transaction.csv
${EVO_CFD_DATA_ROOT}/ieee-fraud-detection/train_identity.csv
```

Then run `clean_data.ipynb` from top to bottom. The notebook exports:

```text
${EVO_CFD_DATA_ROOT}/fraud_corp_ieee_output_2024.csv
${EVO_CFD_DATA_ROOT}/fraud_corp_ieee_output_2024.feature_metadata.json
```

`/data` is the default data root. To use another location, set `EVO_CFD_DATA_ROOT` before running both `clean_data.ipynb` and `run_ieee.sh`. If the raw CSV files are not already extracted, set `IEEE_FRAUD_ZIP` to the zip path; for example `IEEE_FRAUD_ZIP=/data/ieee-fraud-detection.zip`.

## Running EvoCFD

Run the experiment with:
```bash
bash run_ieee.sh
```

By default, `run_ieee.sh` runs `stage_1`. To run all three continual stages:

```bash
STAGE_LIST="stage_1 stage_2 stage_3" bash run_ieee.sh
```


We also provide the scripts for MerFraud and PerFraud datasets, namely `run_MerFraud.sh` and `run_PerFraud.sh`.