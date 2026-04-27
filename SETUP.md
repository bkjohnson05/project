# Setup Instructions

1. Download subset of kaggle data:
go to https://www.kaggle.com/competitions/herbarium-2022-fgvc9/code
create kaggle notebook and run data/herbariumprepv2.ipynb

2. Create environment:
conda create -n herbarium python=3.10
conda activate herbarium

3. Install dependencies:
pip install -r requirements.txt

4. Run experiments:
python -m experiments.run_frozen_baseline
python -m experiments.run_partial_baseline
python -m experiments.run_full_baseline

python -m experiments.run_frozen_aug
python -m experiments.run_partial_aud
python -m experiments.run_full_aug

python -m experiments.run_frozen_aug_wd
python -m experiments.run_partial_aud_wd
python -m experiments.run_full_aug_wd