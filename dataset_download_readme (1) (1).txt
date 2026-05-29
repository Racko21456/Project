SOC Alert Triage Dataset Starter Pack

IMPORTANT:
1. kaggle_json_TEMPLATE_ONLY.json is NOT a real Kaggle token.
2. You must download your real kaggle.json from your own Kaggle account.
3. Go to Kaggle > Settings > API > Create New Token.
4. Upload that real kaggle.json to Google Colab.
5. The sample_soc_alert_dataset.csv file is a small synthetic test dataset only. Use it only to test whether the Python script runs.

Recommended real datasets:
- UNSW-NB15: easiest starting point
- CICIDS 2017: strong IDS dataset
- CSE-CIC-IDS2018: larger dataset, needs more RAM
- NSL-KDD: benchmark dataset

Google Colab setup:
from google.colab import files
files.upload()  # upload your real kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle

Download UNSW-NB15:
!mkdir -p /content/datasets/unsw_nb15
!kaggle datasets download -d mrwellsdavid/unsw-nb15 -p /content/datasets/unsw_nb15 --unzip

Run the project script on UNSW-NB15:
!python soc_alert_triage_project.py --dataset-path /content/datasets/unsw_nb15 --target label --skip-lstm

Run the project script on the sample dataset:
!python soc_alert_triage_project.py --dataset-path sample_soc_alert_dataset.csv --target Label --skip-lstm
