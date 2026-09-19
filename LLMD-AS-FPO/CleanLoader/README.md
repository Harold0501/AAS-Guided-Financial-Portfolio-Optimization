# LLM-driven Algorithm Selection for Financial Portfolio Optimization

This repository contains the experiments and code for the paper **LLM-driven Algorithm Selection for Financial Portfolio Optimization**.

## Setup

All codes are running in `python 3.11` environment.

All dependencies are listed in `requirements.txt`. To install them, simply run:

```bash
pip install -r requirements.txt
```

## Data Loading

All stock assets are stored in `asset_components.xlsx`. You can get data of different assets by:

```bash
python load_data.py 
```

## Target Portfolio caculation

You can get target portfolio by:

```bash
python portfolio_optimization.py -f <asset file name>
```

## Model Training

Once you have created the target portfolio, you can train the model using:

if Linux/macOS (Unix)

```bash
CUDA_VISIBLE_DEVICES=<GPU Number> python experiment.py -a <Asset> -p <Portfolo> -o <Model> -b <Backtest Start> -z sophia
```

if Windows (using cmd)

```bash
set CUDA_VISIBLE_DEVICES=<GPU Number>
python experiment.py -a <Asset> -p <Portfolo> -o <Model> -b <Backtest Start> -z sophia
```

(using powershell)

```bash
$env:CUDA_VISIBLE_DEVICES=<GPU Number>
python experiment.py -a <Asset> -p <Portfolo> -o <Model> -b <Backtest Start> -z sophia
```

`Asset` specifies the asset file.
`Portfolio` indicates the target portfolio type (e.g., Max-Sortino or One-Hot).
`Model` can be LSTM, Transformer, or Mamba.
`-z sophia` sets the optimizer to Sophia.

## Backtest

To get the backtest result of different algorithms, simply run:

```bash
python backtest.py
```
