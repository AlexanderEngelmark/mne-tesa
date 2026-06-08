# mne-tesa

A Python plugin for MNE-Python implementing the TMS-EEG signal analyzer (TESA) preprocessing pipeline. See the [reference paper](https://doi.org/10.1016/j.neuroimage.2016.11.001) by Rogasch et al. (2017) for background.

### Setup

The recommended way to install is with (https://docs.astral.sh/uv/):

Create a virtual environment with:

```bash
uv venv 
```

Then activate the environment:

https://docs.astral.sh/uv/pip/environments/


**For example like this when using bash**

```bash
source .venv/bin/activate
```

### Installing the plugin (Will install MNE-Python and other required libraries)

Clone the repository: 

```bash 
git clone https://codeberg.org/AlexanderEngelmark/mne-tesa.git

cd mne-tesa
```

```bash
uv pip install -e .
```

Or with standard `pip`:

```bash
pip install -e .
```

Or with `conda`:

```bash
conda env create -n mne-tesa python=3.12
conda activate mne-tesa
pip install -e .
```

## Jupyter kernel

To run the tutorial notebook, register a Jupyter kernel for your environment:

**Using uv:**

```bash
uv pip install ipykernel
uv run python -m ipykernel install --user --name=mne-tesa
```

**Using pip (in a virtual environment):**

```bash
pip install ipykernel
python -m ipykernel install --user --name=mne-tesa
```

**Using conda:**

```bash
conda install ipykernel
python -m ipykernel install --user --name=mne-tesa
```

## Example data

The tutorial requires sample BrainVision EEG files. Download them from Zenodo:

```bash
mkdir -p example_data
wget -P example_data https://zenodo.org/records/20598851/files/TMS_EEG64Magventure80.eeg?download=1
wget -P example_data https://zenodo.org/records/20598851/files/TMS_EEG64Magventure80.vhdr?download=1
wget -P example_data https://zenodo.org/records/20598851/files/TMS_EEG64Magventure80.vmrk?download=1
```

## Getting started

Launch Jupyter and open the tutorial:

```bash
jupyter lab tutorial.ipynb
```

Or with the classic notebook interface:

```bash
jupyter notebook tutorial.ipynb
```

This notebook provides a walkthrough of designing a TMS-EEG preprocessing pipeline in MNE-Python.

## Documentation

https://mne-tesa.readthedocs.io/en/latest/index.html
