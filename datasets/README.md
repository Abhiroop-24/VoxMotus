# Dataset Workflow

This project supports two dataset sources:

1. Runtime-collected angle datasets (best for your exact camera setup)
2. External pose/action datasets listed in datasets/registry.json

## 1) Collect your own labeled angle dataset

Enable capture in .env:

APP_COLLECT_TRAINING_DATA=true

Run the app and perform all classes with balanced coverage:

- idle
- standing
- squat
- pushup
- transition

Captured files are written to outputs/datasets as training_angles_*.csv.

## 2) Download external datasets

List supported registry ids:

python scripts/fetch_pose_datasets.py --list

Download one dataset:

python scripts/fetch_pose_datasets.py --dataset coco_keypoints_2017

If a host has SSL chain issues (observed on some UCF mirrors), use:

python scripts/fetch_pose_datasets.py --dataset ucf101_splits --insecure

Download all direct-download entries:

python scripts/fetch_pose_datasets.py --all

For manual datasets, use --include-manual to print links and registration pages.

## Kaggle Downloads

Install is already set in the project venv; to download from Kaggle you need:

1. Kaggle API token at ~/.kaggle/kaggle.json
2. File permission set to 600

Example:

python scripts/fetch_kaggle_dataset.py --dataset uciml/human-activity-recognition-with-smartphones --unzip

## 3) Train deterministic classifier weights

python train_activity_classifier.py --input-glob "outputs/datasets/training_angles_*.csv" --output models/activity_weights.json

For Kaggle landmark CSVs (for example datasets/dataset_all_points.csv with class,x1,y1,...,v33):

python train_activity_classifier.py --input-glob "datasets/dataset_all_points.csv" --output models/activity_weights.json

Deterministic mapping used for this dataset family:
- rest -> idle
- left/right bicep, tricep, shoulder -> standing
- squat/pushup/transition -> direct map when present

If a dataset does not include all runtime classes, missing classes keep stable default priors so inference stays well-defined.

## 4) Use trained weights at runtime

Set in .env:

APP_AI_WEIGHTS_PATH=models/activity_weights.json
APP_AI_CONFIDENCE=0.82

Restart app. The dashboard shows model source and trust status.
