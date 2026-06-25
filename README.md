# Multilayer Mapping UI

<p align="center">
  <img src="https://raw.githubusercontent.com/rmtxcl-hub/Assets/main/Picture%201.png" width="49%">
  <img src="https://raw.githubusercontent.com/rmtxcl-hub/Assets/main/Picture%202.png" width="49%">
</p>

A simple desktop UI for **multilayer thin-film optical mapping** using the **transfer matrix method (TMM)** and a local optical-constants database based on **refractiveindex.info** data.

This tool is designed for quick and easy generation of **wavelength-angle maps** of:

- **Reflectivity** `R`
- **Emissivity** `E = 1 - R`
- **Band-Angle-Averaged Emissivity**
- **Polarization Contrast and Ellipsometry**
- **E and B Field Intensity Maps**

It is intended for practical exploration of:

- multilayer thin films
- angle-dependent optics
- infrared emissivity
- thermal-radiation-related optical design
- planar metamaterial / coating prototyping
- ellispometry applications

---

## Overview

Many TMM-based tools already exist, but they are often code-first or inconvenient for quick visual exploration.

This project focuses on a more accessible workflow:

- simple desktop UI
- direct material selection
- multilayer stack setup without manual scripting
- quick 2D optical mapping
- optional overlay tools for visual analysis
- built-in material-data browsing

The goal is not to replace advanced simulation frameworks, but to make **thin-film optical mapping faster and easier to use**.

---

## Main features

- Desktop GUI built with **PySide6**
- Multilayer thin-film optics using **transfer matrix method**
- Local optical-constants lookup through a database wrapper
- Generates **wavelength-angle maps**
- Supports:
  - **s polarization**
  - **p polarization**
  - **both** (average of s and p)
- Switches between:
  - **Reflectivity**
  - **Emissivity**
  - **Band-Angle-Averaged Emissivity**
  - **Polarization Contrast and Ellipsometry**
  - **E and B Field Intensity Maps**
- Optional overlays:
  - **Contour**
  - **Submask**
  - **Peak**
- Built-in material browser for plotting:
  - `n(λ)`
  - `k(λ)`
- Export figures as:
  - PNG
  - PDF
  - SVG

---

## What the program does

The program has two main tabs.

### 1. Mapping tab

This is the main simulation UI.

You can:

- choose a substrate
- choose the number of layers
- assign a material to each layer
- set layer thicknesses in **µm**
- define wavelength range and sampling
- define angle range and sampling

The program then computes a 2D map over wavelength and incidence angle.

That means you can quickly visualize how a multilayer stack behaves spectrally and angularly, which is useful for:

- emissivity design
- reflectivity analysis
- stack comparison
- thin-film screening

Optional overlays can be enabled to help inspect strong features in the map.

### 2. Data tab

This tab is for browsing material optical data.

For each material key, the program can display:

- refractive index `n`
- extinction coefficient `k`
- wavelength range
- max/min values
- average values over a user-selected wavelength interval

This is useful when selecting candidate materials before building a multilayer stack.

---

## Physics basis

The program uses the **transfer matrix method (TMM)** for planar multilayer structures.

For a given stack, it computes the complex reflection coefficient and then evaluates:

- Reflectivity: `R = |r|^2`
- Emissivity: `E = 1 - R`

For `both` polarization, the displayed result is based on the average of the `s`- and `p`-polarized reflectivities.

So this tool is mainly suited for:

- planar multilayer optical systems
- angle-resolved reflectivity/emissivity studies
- rapid material/thickness exploration

---

## File structure

Main files in this project:

- `UI.py`  
  Main entry point. Run this file to start the app.

- `TMM.py`  
  Transfer matrix method engine.

- `nkwrap.py`  
  Optical-constants lookup and interpolation wrapper.

- `map_collection.py & map_collection2.py`  
  Computes weighted emissivity & ellipsometry parameters

- `map_mod.py`  
  Map-generation logic for reflectivity/emissivity.

- `field_collection.py`  
  Computes field-based parameters

- `contour.py`  
  Contour-style overlay generation.

- `submask.py`  
  Submask extraction from map features.

- `simplepeak.py`  
  1D peak detection helper.

- `material_keys_tagged.txt`  
  Tagged material-key list.

- `data_f.sqlite`  
  Local optical-constants database.

---

## Installation

1. Download this repository.

   * Either download it as a ZIP from GitHub and extract it, or clone it with Git:

2. Download the database file `data_f.sqlite` from the Google Drive link below:

**https://drive.google.com/file/d/1z62c9xKyybTc6KxUo_Ws5GOGtOYs5Mg8/view?usp=sharing**

3. Place `data_f.sqlite` in the same folder as the main Python files.

Your folder should look like this:

```text
Multilayer-Mapping-UI/
├─ UI.py
├─ TMM.py
├─ nkwrap.py
├─ map_mod.py
├─ contour.py
├─ submask.py
├─ simplepeak.py
├─ material_keys_tagged.txt
├─ data_f.sqlite
├─ requirements.txt
└─ README.md
```

### Launch

Run the program with:

```bash
python UI.py
```

This launches the GUI.

### Notes

* `UI.py` is the main entry point of the application.
* `data_f.sqlite` is required for the material database and optical-constants lookup.
* The database file is not included directly in this repository because of file size.
* If `data_f.sqlite` is missing or placed in the wrong folder, the program may not work correctly.

### Requirements

- Python 3.10+ (3.12.x recommended)
- `numpy`
- `scipy`
- `matplotlib`
- `PySide6`

Install dependencies with:

```bash
pip install -r requirements.txt
