# CSPC Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup
Create the environment for a given lab:
```bash
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc
## PW1 Lab B - Data, Plotting, and Automation
- **Data Observation:** The dataset demonstrates exponential radioactive decay over time.
- **Comparison:** The observed experimental data closely matches the theoretical analytical decay curve.
- **Automation:** The Snakemake pipeline automates figure generation, rebuilding `figure.png` only when source data or scripts change.