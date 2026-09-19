# PG-VAE-LSTM

**Physics-Guided Variational Autoencoder–Long Short-Term Memory for Multi-Source Evapotranspiration Fusion**

To address uncertainty and physical inconsistency in multi-source evapotranspiration (ET) fusion, we propose **PG-VAE-LSTM**, a physics-guided probabilistic temporal fusion framework. The model uses LSTM to capture temporal dependencies, VAE to represent uncertainty among heterogeneous ET products in a latent probabilistic space, and energy-, water-, and non-negativity constraints to improve the physical consistency of fused ET estimates.

The repository also includes LSTM, ED-LSTM, and VAE-LSTM for ablation experiments.

### Dataset

The dataset used in this study is based on the multi-source ET dataset compiled and reconstructed by Xu et al. (2025). It contains 13 reconstructed global ET products, ERA5-Land meteorological and land-surface variables, ancillary land-surface data, and ET observations from 462 global eddy-covariance flux-tower sites.The final AutoML-fused ET product from Xu et al. (2025) is not used as an input to PG-VAE-LSTM.

The code expects preprocessed `train`, `val`, and `test` datasets containing:

- `X.npy`: input sequences (`N × 365 × 39`)
- `y.npy`: observed ET
- `site_id.npy`: flux-tower site IDs
- `energy_available_wm2.npy`: available-energy constraint
- `water_available_mm_day.npy`: water-availability constraint

Reference:

Xu, Q., Li, L., Wei, Z., et al. (2025). A multimodal machine learning fused global 0.1° daily evapotranspiration dataset from 1950–2022. *Agricultural and Forest Meteorology*, 372, 110645.

### Requirements

The code requires **Python 3.10 or later** and the following packages:

- PyTorch
- NumPy
- pandas
- SciPy
- Matplotlib
- tqdm

### Prepare Config File

All model, data, training, loss, and evaluation settings are defined in `config.py`.

The default PG-VAE-LSTM configuration uses a 365-day input sequence, hidden size of 256, latent size of 32, dropout rate of 0.15, batch size of 64, learning rate of 0.001, and 1000 training epochs.

Parameters can also be changed through command-line arguments.

### Train Model

Run the following command in the repository directory to train PG-VAE-LSTM:

```bash
python train.py --modelname PG-VAE-LSTM
```

The ablation models can be trained by changing `--modelname` to `LSTM`, `ED-LSTM`, or `VAE-LSTM`.

### Model Evaluation

Run the following command to evaluate the trained model:

```bash
python evaluate.py --modelname PG-VAE-LSTM
```

The evaluation script reports pooled and site-level R², KGE, RMSE, and Bias, and saves the test predictions for further analysis.

### Detailed Analysis

Run the following command to generate the density scatter plot from the saved test predictions:

```bash
python plot_density_scatter.py outputs/pg_vae_lstm/predictions_test.npz --out density_scatter.png
```
