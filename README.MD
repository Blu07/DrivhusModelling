# Drivhus Modelling

This project analyses measured data from a greeenhouse and attempts to calculate the greenhouse's effective heat capacity (C). The calculations are based on solar irradiance and temeperature differences inside and outside.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

```bash
# Clone the repository
git clone https://github.com/Blu07/DrivhusModelling.git

# Navigate to the project directory
cd DrivhusModelling

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install requirements.txt

# Run the program
python3 main.py
```

## Usage

### Editing settings
Change the following parameters to edit which days are included in the analysis. Only these days will be plotted 
```python
./utils/settingsAndConstants.py


#%% Settings for the analysis
# The days to include in the analysis
start_day = 1
num_days = 100
```

#

Toggle True/False to choose which functions are run.
```python
./main.py

# Bytt mellom True/False for å velge å gjøre/ikke gjøre følgende

if False: days_data = searchDaysForU(days)
else:    days_data = generateCListFromFile(days)

if True: printParams(days_data)
if True: plotDrivhus(drivhus_data, save_fig=False)
if True: plotCPerDay(days_data, save_fig=False)
if True: showPlots()
```

