# Installation Guide

This guide walks through setting up SWIM-RS from scratch, including dependencies and optional components.

## Prerequisites

- **Python 3.13** (required)
- **conda** (recommended; see [Non-Conda Installation](#non-conda-installation) below for alternatives)
- **Git** (for installing from source)

## Conda Installation (Recommended)

### 1. Install Conda (if needed)

If you don't have conda installed, we recommend Miniconda (lightweight) or Miniforge (community-driven, conda-forge default).

### Option A: Miniconda

Download from: https://docs.conda.io/en/latest/miniconda.html

**Windows:**
1. Download `Miniconda3-latest-Windows-x86_64.exe` from the link above
2. Run the installer and follow prompts
3. Open "Anaconda Prompt" from the Start menu for conda commands

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**macOS (Intel):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**macOS (Apple Silicon):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### Option B: Miniforge (recommended for Apple Silicon and conda-forge defaults)

Download from: https://github.com/conda-forge/miniforge

**Windows:**
1. Download `Miniforge3-Windows-x86_64.exe` from the releases page
2. Run the installer and follow prompts
3. Open "Miniforge Prompt" from the Start menu

**Linux:**
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

**macOS (Apple Silicon):**
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

After installation:
- **Windows**: Open Anaconda Prompt or Miniforge Prompt
- **Linux/macOS**: Restart your terminal or run `source ~/.bashrc` (or `~/.zshrc`)

### 2. Create a Conda Environment

Create a dedicated environment for SWIM-RS. These commands work the same on Windows (Anaconda Prompt), macOS, and Linux:

```bash
conda create -n swim python=3.13 -y
conda activate swim
```

### 3. Install PEST++ (for calibration)

PEST++ is required for parameter estimation. Install from conda-forge:

```bash
conda install -c conda-forge pestpp -y
```

Verify installation:
```bash
pestpp-ies --version
```

You should see version output like `PEST++ Version 5.x.x`.

### 4. Install Geospatial Dependencies

Some geospatial libraries install more reliably via conda:

```bash
conda install -c conda-forge geopandas rasterio pyproj fiona shapely -y
```

### 5. Install SWIM-RS

**From PyPI (stable release):**
```bash
pip install swimrs
```

**From GitHub (latest):**
```bash
pip install git+https://github.com/dgketchum/swim-rs.git
```

**Editable install (for development):**
```bash
git clone https://github.com/dgketchum/swim-rs.git
cd swim-rs
pip install -e .
```

**With OpenET models (optional):**

To use OpenET algorithm implementations directly (for custom ETf calculations):

```bash
pip install swimrs[openet]
```

Or if installing from source:
```bash
pip install -e ".[openet]"
```

### 6. Set Up Google Earth Engine (optional)

Earth Engine access is only needed if you want to extract fresh remote sensing data. The shipped examples include pre-extracted data.

1. Go to https://earthengine.google.com/ and sign up
2. Run `earthengine authenticate` and complete OAuth in browser
3. Verify with: `python -c "import ee; ee.Initialize(); print('EE OK')"`

### 7. Verify Installation

```bash
swim --help
pestpp-ies --version
```

---

## Non-Conda Installation

If you prefer not to use conda, you can install SWIM-RS with pip and download PEST++ executables directly.

### 1. Create a Virtual Environment

**Linux/macOS:**
```bash
python3.13 -m venv ~/.venvs/swim
source ~/.venvs/swim/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv $env:USERPROFILE\.venvs\swim
& $env:USERPROFILE\.venvs\swim\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```batch
python -m venv %USERPROFILE%\.venvs\swim
%USERPROFILE%\.venvs\swim\Scripts\activate.bat
```

### 2. Install SWIM-RS via pip

```bash
pip install --upgrade pip
pip install swimrs
```

Or from source:
```bash
git clone https://github.com/dgketchum/swim-rs.git
cd swim-rs
pip install -e .
```

### 3. Install PEST++ Executables

Download pre-built PEST++ binaries from the official releases:

**Download:** https://github.com/usgs/pestpp/releases

1. Download the appropriate archive for your platform:
   - Windows: `pestpp-X.X.X-win.zip`
   - Linux: `pestpp-X.X.X-linux.tar.gz`
   - macOS: `pestpp-X.X.X-mac.tar.gz`

2. Extract and add to PATH:

**Linux/macOS:**
```bash
# Example for Linux
wget https://github.com/usgs/pestpp/releases/download/X.X.X/pestpp-X.X.X-linux.tar.gz
tar -xzf pestpp-X.X.X-linux.tar.gz
sudo mv pestpp-X.X.X-linux/bin/* /usr/local/bin/
# Or add to PATH in ~/.bashrc:
# export PATH="$HOME/pestpp-X.X.X-linux/bin:$PATH"
```

**Windows:**
1. Extract the zip to a folder (e.g., `C:\pestpp`)
2. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit `Path` under User variables
   - Add `C:\pestpp\bin`
3. Restart your terminal

**Verify:**
```bash
pestpp-ies --version
```

### 4. Install Geospatial Dependencies

Without conda, you may need system libraries for GDAL:

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)
pip install geopandas rasterio fiona pyproj shapely
```

**macOS (Homebrew):**
```bash
brew install gdal
pip install geopandas rasterio fiona pyproj shapely
```

**Windows:**

Use pre-built wheels from Christoph Gohlke's archive or try:
```bash
pip install pipwin
pipwin install gdal fiona rasterio
pip install geopandas pyproj shapely
```

Or use the [OSGeo4W installer](https://trac.osgeo.org/osgeo4w/) which includes GDAL and Python bindings.

### 5. Verify Installation

```bash
swim --help
pestpp-ies --version
python -c "import geopandas; print('geopandas OK')"
```

---

## Complete Setup Scripts

### Conda (Linux/macOS)

```bash
#!/bin/bash
conda create -n swim python=3.13 -y && conda activate swim
conda install -c conda-forge pestpp geopandas rasterio pyproj fiona shapely -y
git clone https://github.com/dgketchum/swim-rs.git && cd swim-rs
pip install -e .
swim --help && pestpp-ies --version
```

### Conda (Windows — Anaconda Prompt)

```batch
conda create -n swim python=3.13 -y && conda activate swim
conda install -c conda-forge pestpp geopandas rasterio pyproj fiona shapely -y
git clone https://github.com/dgketchum/swim-rs.git && cd swim-rs
pip install -e .
swim --help && pestpp-ies --version
```

### Non-Conda (Linux/macOS)

```bash
#!/bin/bash
python3.13 -m venv ~/.venvs/swim && source ~/.venvs/swim/bin/activate
pip install --upgrade pip

# Install GDAL system library first (Ubuntu: sudo apt-get install gdal-bin libgdal-dev)
pip install geopandas rasterio fiona pyproj shapely

# Install swim-rs
git clone https://github.com/dgketchum/swim-rs.git && cd swim-rs
pip install -e .

# Download PEST++ from https://github.com/usgs/pestpp/releases and add to PATH
swim --help && pestpp-ies --version
```

## Troubleshooting

### GDAL/Fiona errors

If you see GDAL-related errors, ensure geospatial libs are from conda:
```bash
conda install -c conda-forge gdal fiona rasterio --force-reinstall -y
```

### PEST++ not found

Ensure conda-forge channel is available:
```bash
conda config --add channels conda-forge
conda install pestpp -y
```

### Earth Engine authentication fails

Try re-authenticating with a fresh token:
```bash
earthengine authenticate --force
```

### Import errors after pip install

Ensure you're in the correct conda environment:
```bash
conda activate swim
which python  # Linux/macOS - should point to your conda env
where python  # Windows - should point to your conda env
```

### Apple Silicon (M1/M2) issues

Use Miniforge instead of Miniconda for better ARM64 support:
```bash
conda install -c conda-forge numpy scipy pandas --force-reinstall -y
```

### Windows: "swim" command not found

If `swim` isn't recognized after installation, ensure you're in the Anaconda/Miniforge Prompt (not regular Command Prompt or PowerShell). Alternatively, use:
```batch
python -m swimrs.cli --help
```

### Windows: Long path errors

Enable long paths in Windows 10/11:
1. Open Group Policy Editor (`gpedit.msc`)
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Enable "Enable Win32 long paths"

Or via Registry:
```batch
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### Windows: Git not found

Install Git for Windows from: https://git-scm.com/download/win

Or install via conda:
```bash
conda install -c conda-forge git -y
```

## Environment File (optional)

For reproducible environments, create `environment.yml`:

```yaml
name: swim
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - pestpp
  - geopandas
  - rasterio
  - pyproj
  - fiona
  - shapely
  - numpy
  - scipy
  - pandas
  - xarray
  - zarr
  - pip
  - pip:
    - swimrs
```

Install with:
```bash
conda env create -f environment.yml
conda activate swim
```

## Next Steps

- Run the [Quick Start](../README.md#quick-start-fort-peck-montana-full-workflow) example
- Explore the [Examples](../README.md#examples) for tutorials and workflows
- Read the [Data Extraction Guide](data_extraction.md) if you need to pull fresh EE data
