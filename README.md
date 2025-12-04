# dftfe-phonons

A Python utility that connects **Phonopy** and **DFT-FE** workflows.
This script automates:

1. **Preparing DFT-FE inputs** for each displaced supercell (`domainBoundingVectors.inp` and `coordinates.inp`)  
2. **Reading DFT-FE forces** and assembling a valid **Phonopy `FORCE_SETS`** file

Designed for large-scale phonon calculations using **DFT-FE** as the DFT backend and **Phonopy** for post-processing.

---

## Features

### ✅ Prepare DFT-FE Inputs  
For each `POSCAR-*` displaced structure:
- Converts VASP POSCAR to DFT-FE format  
- Writes:
  - `domainBoundingVectors.inp` (in **Bohr**)  
  - `coordinates.inp` (fractional positions, valence info, optional magnetism)  
- Supports interactive or JSON-driven valence electron specification  
- Supports magnetic or non-magnetic calculations  
- Works with **PhonopyAtoms** from old and new Phonopy versions

---

### ✅ Generate Phonopy `FORCE_SETS`  
Given:
- Reference structure (`SPOSCAR`)
- Per-displacement DFT-FE force files (`forces.txt`)

The script:
- Detects the displaced atom  
- Converts forces to **eV/Å**  
- Writes **Type-1 FORCE_SETS** compatible with Phonopy  
- Handles Ha/Bohr → eV/Å conversion automatically  

---

## Installation

### Requirements

- Python 3.8+  
- `phonopy` ≥ 2.3  
- `numpy`  

Install dependencies:

```bash
pip install phonopy numpy
```

Clone this repository:

```bash
git clone https://github.com/<your-username>/dftfe-phonons.git
cd dftfe-phonons
```

---

## Usage

The script provides two modes:

---

# 1. Prepare DFT-FE Inputs

```bash
python dftfe-phonons.py --prepare_input
```

This mode:
- Scans for `POSCAR-*` files
- Detects all unique elements
- Prompts for **valence electrons**
- Prompts for **magnetism** settings  
- Generates folders:

```
dftfe_in/POSCAR-001/
dftfe_in/POSCAR-002/
...
```

Each contains:

```
domainBoundingVectors.inp
coordinates.inp
```

### Optional arguments

| Flag | Description |
|------|-------------|
| `--zvalence-json file.json` | Use a JSON mapping instead of interactive input |
| `--magnetic --magvalue X` | Non-interactive global magnetization |
| `--poscar-glob "POSCAR-*"` | Custom glob for displaced structures |
| `--out-root name` | Base directory for output |

### Example JSON file
```json
{
  "Fe": [26, 8],
  "C": [6, 4]
}
```

---

# 2. Build FORCE_SETS from DFT-FE Forces

```bash
python dftfe-phonons.py --read_force --reference SPOSCAR
```

### What it does
- Reads the reference supercell (`SPOSCAR`)
- Reads each displaced `POSCAR-*`
- Detects which atom moved
- Reads forces from:

```
dftfe_in/POSCAR-001/forces.txt
```

- Converts units
- Writes a complete `FORCE_SETS` file

### Optional arguments

| Flag | Description |
|------|-------------|
| `--forces-filename` | Default: `forces.txt` |
| `--force-unit` | `Ha/Bohr` (default) or `eV/Ang` |
| `--output` | Output FORCE_SETS filename |
| `--poscar-glob` | Pattern for displaced POSCARs |

Example:

```bash
python dftfe-phonons.py --read_force     --reference SPOSCAR     --force-unit Ha/Bohr     --output FORCE_SETS
```

---

## Input File Layout

Your directory should look like:

```
SPOSCAR
POSCAR-001
POSCAR-002
...
dftfe-phonons.py
```

After `--prepare_input`, you get:

```
dftfe_in/
    POSCAR-001/
        domainBoundingVectors.inp
        coordinates.inp
    POSCAR-002/
        ...
```

Run your DFT-FE calculations inside each folder, producing `forces.txt`.

---
