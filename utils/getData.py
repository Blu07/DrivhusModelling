import numpy as np
import pandas as pd

from utils.settingsAndConstants import LUX_CONVERSION



def cleanData(data, lower, upper, errors=[], calibration=0, factor=1, lower_derivative=None, upper_derivative=None):
    cleansed_data = []
    for i, value in enumerate(data):
        invalid = \
            (value in errors) or \
            value < lower or \
            value > upper or \
            np.isnan(value)
            
        # Add derivative thresholds
        if lower_derivative is not None and i < len(data) - 2:
            if abs(np.nan_to_num(data[i+1] - value)) < lower_derivative:
                invalid = True
                
        if upper_derivative is not None and i < len(data) - 2:
            if abs(np.nan_to_num(data[i+1] - value)) > upper_derivative:
                invalid = True
        
        cleansed_data.append(np.nan if invalid else (value + calibration) * factor)

    return cleansed_data



def weightFilterData(data, weights=5, rate=0.5):
    """
    Filtrerer data ved å bruke en normalistert geometrisk vekting.

    Args:
        data (list): Liste med numeriske verdier.
        weights (int): Antall vekter som brukes.
        rate (float): Rate for endring i geometrisk vekting.

    Returns:
        filtered_data (list): Filtrert data.
    """
    # Lag en normalisert geometrisk vekting
    weight_values = [(1 - rate) * rate**j for j in range(weights)]
    normalization_factor = sum(weight_values)

    weight_values = [w / normalization_factor for w in weight_values]  # Normaliser vektene

    filtered_data = data[:]  # Kopier for å ikke endre originalen

    for i in range(1, len(data) - weights):
        summed = sum(data[i + j] * weight_values[j] for j in range(weights))
        filtered_data[i] = summed

    return filtered_data





def cleanAllData(d_data: pd.DataFrame):
    
    time_calibration = 3600 # GMT+1
    
    d_data['time'] = cleanData(
        pd.to_numeric(d_data['time'].astype(int)),
        lower=1700000000,
        upper=1900000000,
        errors=[],
        calibration=time_calibration # Adjust for GMT+1
    )

    d_data['tempIn'] = cleanData(
        pd.to_numeric(d_data['tempIn'].astype(float)),
        lower=-25,
        upper=40,
        errors=[],
        calibration=0
    )

    d_data['tempOut'] = cleanData(
        pd.to_numeric(d_data['tempOut'].astype(float)),
        lower=-25,
        upper=40,
        errors=[-127, 85],
        calibration=0
    )

    d_data['batVolt'] = cleanData(
        pd.to_numeric(d_data['batVolt'].astype(float)),
        lower=2.0,
        upper=4.5,
        errors=[],
        calibration=0
    )

    d_data['light'] = cleanData(
        pd.to_numeric(d_data['light'].astype(int)),
        lower=0,
        upper=100_000,
        errors=[],
        calibration=0,
        factor=LUX_CONVERSION
    )

    
    # Drop rows where time is nan. These rows can not be used.
    d_data = d_data.dropna(subset=['time'])

    return d_data



def combineFromIntervals(d_data):
    """ Combine every set of 5 readings per 4 minutes into the average of those readings."""
    combined_data = []
    
    sums = np.array([0, 0, 0, 0], dtype=float)
    nums = np.array([0, 0, 0, 0], dtype=int)
    
    
    prev_time = d_data["time"].iloc[0]
    prev_skip_time = 0
    wait_time = 240 # 4 minutes # there are 5 readings per 5 minutes, but use 4 to make sure all are included
    

    for id, time, *values in d_data.itertuples():
        
        # Row does not have a valid time, skip
        if np.isnan(time):
            continue
        
        # all values in row are NaN, skip 
        if not np.any(values):
            continue
        
        # Accumulated values
        for i, v in enumerate(values):
            
            if not np.isnan(v):
                sums[i] += v
                nums[i] += 1
            
           
        # Find the average of the accumulated data
        if time > prev_skip_time + wait_time:
            
            avgs = sums/nums
            combined_data.append([prev_time, *avgs])
            
            prev_skip_time = time
            
            # Reset sums and nums
            sums *= 0
            nums *= 0
        
        
        prev_time = time
    
    
    return pd.DataFrame(combined_data, columns=["time", "tempIn", "tempOut", "light", "batVolt"])


def interpolateDData(d_data: pd.DataFrame, resolution):
    
    time = np.linspace(d_data['time'].iloc[0], d_data['time'].iloc[-1], resolution)
    
    new_data = pd.DataFrame({
        "time": time,
        "tempIn": np.interp(time, d_data['time'], d_data['tempIn']),
        "tempOut": np.interp(time, d_data['time'], d_data['tempOut']),
        "light": np.interp(time, d_data['time'], d_data['light']),
        "batVolt": np.interp(time, d_data['time'], d_data['batVolt'])
    })
    
    return new_data



def addGroundTemp(d_data):
    
    p = 86400 # seconds in a day
    p_seasonal = 365 * p # seconds in a year
    x = d_data['time']
    # Daily variation + seasonal variation + linear trend + constant offset
    tempGround = 3 * np.sin(2 * np.pi / p * (x - p/2)) \
                    + 10 * np.sin(2 * np.pi / p_seasonal * (x-90)) \
                    + (x - x.iloc[0])/p/8 \
                    + 2
    
    d_data['tempGround'] = tempGround
    
    return d_data


# FETCH DRIVHUS DATA AND CLEAN FAULTY READINGS
def getDrivhusData(fileName: str = "drivhus.txt", **kwargs):

    print("Henter data for Drivhus")
    d_data = pd.read_csv(fileName, **kwargs)
    d_data.columns = ["time", "tempIn", "tempOut", "light", "batVolt"]
    
    # Rense og behandle rå data
    d_data = cleanAllData(d_data)
    d_data = combineFromIntervals(d_data)
    
    resolution = len(d_data) * 5 # Øker gjennomsnitllig oppløsning med 5x
    d_data = interpolateDData(d_data, resolution)
    
    d_data = addGroundTemp(d_data)
    
    return d_data




