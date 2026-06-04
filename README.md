# AdaptiveFL: Concept Drift Detection for Federated Learning

A novel federated learning framework that integrates per-client ADWIN-based 
concept drift detection with dynamic client weighting during aggregation.

## Key Result
AdaptiveFL maintains **97.8% accuracy** under 60% client drift, compared to 
**93.1% for standard FedAvg** — a 4.7 percentage point improvement.

## Project Structure
- src/client/     — FL client with ADWIN drift detector
- src/server/     — Weighted FedAvg aggregation server
- src/data/       — MNIST loader and drift simulator
- experiments/    — 4 experiment scripts
- results/        — Generated plots and CSV results

## Setup
pip install torch torchvision numpy matplotlib pandas scikit-learn river tqdm

## Run Experiments
python experiments/run_experiment1.py
python experiments/run_experiment2.py
python experiments/run_experiment3.py
python experiments/run_experiment4.py

## Paper
Published as IEEE conference paper.
Full paper available in the repository.

## Author
Atharv Dorle — SRM Institute of Science and Technology
