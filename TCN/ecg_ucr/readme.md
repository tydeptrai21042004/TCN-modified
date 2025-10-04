# ECG200 (UCR) — Time Series Classification with TCN

This task adds the **ECG200** dataset from the **UCR Time Series Classification Archive**.  
It is a binary classification problem (*normal* vs *myocardial infarction*) with univariate series of length 96.  
We use `sktime.datasets.load_UCR_UEA_dataset` to auto-download and load the data.

- ECG200 description (classes & heartbeat nature) – Time Series Classification site.  
- `load_UCR_UEA_dataset` API – returns NumPy 3D `(N, C, T)` with `return_type="numpy3D"`.

## How to run

Install dependencies (Colab-friendly):
```bash
pip install -q sktime
