
## Environment Setup & Installation

This project requires geospatial libraries (`GDAL`, `GEOS`, `PROJ`) and `PyTorch`. We manage dependencies using **Miniforge** with the `conda-forge` channel to ensure binary compatibility across platforms [1, 2].

### 1. Prerequisites

If you do not have Conda installed, install **Miniforge3** for your system [1]:

- **macOS (Apple Silicon M1/M2/M3/M4):**

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

- **macOS (Intel x86_64):**

```bash 
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" 
Miniforge3-MacOSX-x86_64.sh
```
- **Linux (x86_64):**

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
Miniforge3-Linux-x86_64.sh
```

- **Windows:** Download and execute the installer from the [Miniforge Repository](https://github.com/conda-forge/miniforge).

---

### 2. Configure Conda Channel Priority

Set channel priority to `conda-forge` to prevent downloading from commercial repositories with rate limits [1]:

```bash 
conda config --add channels conda-forge
conda config --set channel_priority strict
```

---

### 3. Create and Activate the Environment

Create the environment using the repository's `environment.yml` specification [2, 3]:

```bash
conda env create -f environment.yml
conda activate chaos_ml_env
```