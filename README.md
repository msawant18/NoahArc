# NoahArc
This is a production-ready deep learning pipeline for flood detection from Sentinel-1 SAR imagery. 

---

## Overview

NoahArc performs:

- **Binary flood mapping** on 512×512 Sentinel-1 SAR tiles  
- **Explainable AI outputs**: calibrated confidence maps  
- **Georeferenced mask export** compatible with GIS tools  
- Fully **containerized** for reproducible deployment  

Ideal for climate monitoring, disaster response, and research.

---







It performs binary semantic segmentation (flooded vs. non-flooded land), outputs explainable masks with calibrated confidence, and generates georeferenced layers for emergency response and climate risk analytics. 
NoahArc/
├── README.md                # Project overview (2-page compressed)
├── LICENSE                  # Apache 2.0 (or your choice)
├── .gitignore               # Ignore data, logs, build artifacts
├── docker-compose.yml       # Orchestration for full pipeline
├── Dockerfile               # Base image for training
│
├── configs/                 # YAML/JSON configs for training & inference
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── pipeline.yaml
│
├── data/                    # (gitignored) Local data store
│   ├── raw/                 # Sentinel-1 tiles
│   ├── processed/           # Preprocessed inputs
│   └── outputs/             # Predictions, GeoTIFFs
│
├── notebooks/               # Jupyter/Colab notebooks for EDA, demos
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_inference_demo.ipynb
│
├── src/                     # Core pipeline code
│   ├── __init__.py
│   ├── data/                # Data ingestion + preprocessing
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── models/              # Model architectures & training
│   │   ├── unet.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── explainability/      # Grad-CAM, SHAP, visualization
│   │   ├── gradcam.py
│   │   └── shap_utils.py
│   ├── evaluation/          # Metrics, calibration
│   │   ├── metrics.py
│   │   └── calibration.py
│   └── export/              # GeoTIFF/COG export
│       ├── geotiff_writer.py
│       └── postprocess.py
│
├── tests/                   # Unit & integration tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_export.py
│
├── scripts/                 # CLI wrappers & automation
│   ├── run_preprocessing.sh
│   ├── train_model.sh
│   └── run_inference.sh
│
├── docs/                    # Documentation, guides, diagrams
│   ├── model_card.md
│   ├── dataset_card.md
│   └── architecture.png
│
└── requirements.txt         # Python dependencies (or environment.yml)
