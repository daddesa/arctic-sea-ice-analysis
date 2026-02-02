# arctic-sea-ice-analysis
**Statistical Analysis of Arctic Sea Ice Area Using NSIDC Data (1979-2025)** for the *Physics of Energy and Environment* exam (A.Y. 2025/26), University of Rome Tor Vergata

## Project Overview
This repository contains the Python source code used for the statistical analysis presented in the course's final report. 

The script performs an independent analysis of the Sea Ice Area (SIA) September minimums to validate CMIP6 projections. It includes:
- **Data Preprocessing:** Filtering and loading of NSIDC Sea Ice Index data.
- **Trend Analysis:** Comparison between Linear and Quadratic regression models
- **Stability Analysis:** Assessment of system stability via Rolling Window Analysis

## Repository Structure
- `arctic-sea-ice-analysis.py`: The main Python script
- `N_09_extent_v4.0.csv`: The raw dataset provided by NSIDC (September Sea Ice Index).
- `requirements.txt`: List of Python libraries required to run the code.

## How to Run
To reproduce the analysis and generate the plots:

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Run the script**
   ```bash
   python arctic-sea-ice-analysis
### Data provided by the National Snow and Ice Data Center (NSIDC): [https://nsidc.org/data/g02135/versions/4](https://nsidc.org/data/g02135/versions/4)

#### Disclaimer
This project was developed for educational purposes as part of the Physics of Energy and Environment exam. The analysis and code are provided "as is".
