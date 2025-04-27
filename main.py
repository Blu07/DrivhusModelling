# %% Import libraries
print("Importerer Biblioteker")

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from utils.getData import * 



# Measured values from Drivhus
A_AIR = 42 # m^2
A_EKSP = 15 # m^2
A_FLOOR = 15 # m^2

ALBEDO = 0.4
TRANSMISSION = 0.8


LUX_CONVERSION = 1/7.5 # W/m^2 per Lx



#%% SUN CALCULATIONS


def getSunAngleDeg(unixTime: int, lat, lon, d=None):
    dateTime = dt.fromtimestamp(unixTime)
    
    # Local time in minutes
    LT = dateTime.hour * 60 + dateTime.minute

    # Local Standard Time Meridian (LSTM)
    LSTM = 15 * 1  # 15 degrees per hour for UTC+1

    # Day of the year
    if d is None:
        d = dateTime.timetuple().tm_yday
    #print(f"Day of year: {d}")

    # Solar declination angle (δ)
    B = np.radians((360 / 365) * (d - 81))
    #print(f"B (in radians): {B}")
    decl = np.radians(23.44) * np.sin(B)

    # Equation of Time (EOT)
    EOT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    #print(f"EOT (in minutes): {EOT}")

    # Time Correction Factor (TC)
    TC = 4 * (lon - LSTM) + EOT

    # Local Solar Time (LST)
    LST = LT + TC
    HRA = 15 * ((LST / 60) - 12)  # Hour Angle in degrees

    # Solar Elevation Angle
    r = np.arcsin(np.sin(np.radians(lat)) * np.sin(decl) +
                  np.cos(np.radians(lat)) * np.cos(decl) * np.cos(np.radians(HRA)))
    
    return r

getInsolation = lambda sunAngle: 0 if sunAngle < 0 else 1361 * np.sin(sunAngle)  # W/m^2 https://no.wikipedia.org/wiki/Solkonstanten


def getFirstSunRiseTime(day, timeList, lat, lon):
    for t in timeList:
        sunAngle = getSunAngleDeg(t, lat, lon, day)

        if sunAngle > 0:
            return t



#%% CLEANING DATA


def cleanData(data, lower, upper, errors=[], calibration=0, factor=1, lowerDerivative=None, upperDerivative=None):
    cleansedData = []
    for i, value in enumerate(data):
        invalid = \
            (value in errors) or \
            value < lower or \
            value > upper or \
            np.isnan(value)
            
        # Add derivative thresholds
        if lowerDerivative is not None and i < len(data) - 2:
            if abs(np.nan_to_num(data[i+1] - value)) < lowerDerivative:
                invalid = True
                
        if upperDerivative is not None and i < len(data) - 2:
            if abs(np.nan_to_num(data[i+1] - value)) > upperDerivative:
                invalid = True
        
        cleansedData.append(np.nan if invalid else (value + calibration) * factor)

    return cleansedData



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





def cleanAllData(dData: pd.DataFrame):
    
    timeCalibration = 3600 # GMT+1
    
    dData['time'] = cleanData(
        pd.to_numeric(dData['time'].astype(int)),
        lower=1742968800, # 26/04/2025 kl. 07:00
        upper=1900000000,
        errors=[],
        calibration=timeCalibration # Adjust for GMT+1
    )

    dData['tempIn'] = cleanData(
        pd.to_numeric(dData['tempIn'].astype(float)),
        lower=-25,
        upper=40,
        errors=[],
        calibration=0
    )

    dData['tempOut'] = cleanData(
        pd.to_numeric(dData['tempOut'].astype(float)),
        lower=-25,
        upper=40,
        errors=[-127, 85],
        calibration=0
    )

    dData['batVolt'] = cleanData(
        pd.to_numeric(dData['batVolt'].astype(float)),
        lower=2.0,
        upper=4.5,
        errors=[],
        calibration=0
    )

    dData['light'] = cleanData(
        pd.to_numeric(dData['light'].astype(int)),
        lower=0,
        upper=100_000,
        errors=[],
        calibration=0,
        factor=LUX_CONVERSION
    )

    
    # Drop rows where time is nan. These rows can not be used.
    dData = dData.dropna(subset=['time'])

    return dData



def combineFromIntervals(dData):
    """ Combine every set of 5 readings per 4 minutes into the average of those readings."""
    combinedData = []
    
    sums = np.array([0, 0, 0, 0], dtype=float)
    nums = np.array([0, 0, 0, 0], dtype=int)
    
    
    prevTime = dData["time"].iloc[0]
    prevSkipTime = 0
    waitTime = 240 # 4 minutes
    

    for id, time, *values in dData.itertuples():
        
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
        if time > prevSkipTime + waitTime:
            
            avgs = sums/nums
            combinedData.append([prevTime, *avgs])
            
            prevSkipTime = time
            
            # Reset sums and nums
            sums *= 0
            nums *= 0
        
        
        prevTime = time
    
    
    return pd.DataFrame(combinedData, columns=["time", "tempIn", "tempOut", "light", "batVolt"])


def interpolateData(dData: pd.DataFrame, resolution):
    
    time = np.linspace(dData['time'].iloc[0], dData['time'].iloc[-1], resolution)
    
    newData = pd.DataFrame({
        "time": time,
        "tempIn": np.interp(time, dData['time'], dData['tempIn']),
        "tempOut": np.interp(time, dData['time'], dData['tempOut']),
        "light": np.interp(time, dData['time'], dData['light']),
        "batVolt": np.interp(time, dData['time'], dData['batVolt'])
    })
    
    return newData



def addGroundTemp(dData):
    
    p = 86400 # seconds in a day
    x = dData['time']
    tempGround = 3 * np.sin(2 * np.pi / p * (x - p/2)) + (x - x.iloc[0])/p/2 + 10
    
    dData['tempGround'] = tempGround
    
    return dData


# FETCH DRIVHUS DATA AND CLEAN FAULTY READINGS
def drivhusData(fileName: str = "drivhus.txt", **kwargs):

    print("Henter data for Drivhus")
    dData = pd.read_csv(fileName, **kwargs)
    dData.columns = ["time", "tempIn", "tempOut", "light", "batVolt"]
    
    # Rense og behandle rå data
    dData = cleanAllData(dData)
    dData = combineFromIntervals(dData)
    
    resolution = len(dData) * 5 # Øker gjennomsnitllig oppløsning med 5x
    dData = interpolateData(dData, resolution)
    
    dData = addGroundTemp(dData)
    
    return dData





#%% DATA ANALYSIS


def evaluate_params(dData, h_air, h_ground):
    cList = calculateCWithParams(dData, h_air, h_ground)
    c = np.nanmedian(cList)    
    std = np.nanstd(cList)
    
    return c, std, h_air, h_ground, cList
    

def find_best_h(dData, h_air_range, h_ground_range):
    bestSTD = 1e10
    best_params = (None, None)
    best_cList = None

    combos = [(h_air, h_ground) for h_air in h_air_range for h_ground in h_ground_range]
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(evaluate_params, dData, h_air, h_ground) for h_air, h_ground in combos]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"{i+1} / {len(futures)}")

            if result is None:
                print("No result")
                continue
            
            
            c, std, h_air, h_ground, cList = result
            
            if c > 0 and std < bestSTD:
                bestSTD = std
                best_params = (h_air, h_ground)
                best_cList = cList

    return best_params, best_cList


def calculateCWithParams(dData: pd.DataFrame, h_air: float, h_ground: float):
    tempDiff_air = dData['tempIn'] - dData['tempOut']
    tempDiff_ground = dData['tempIn'] - dData['tempGround'] # Ground temp approx
    
    cList = []
    
    resetValues = True
    
    for i in range(0, len(dData)):
        
        if i+1 >= len(dData) or i-1 < 0:
            cList.append(np.nan)
            continue
        
        if resetValues:
            dQ = 0
            countedNoLight = 0
            initTemp = dData['tempIn'].iloc[i-1]
            
            resetValues = False
        
        
        # Count the number of consecutive data points with low light
        if dData['light'].iloc[i] < 3:
            countedNoLight += 1
        
            
        # Reset values when the sun sets (with 3 W/m^2 threshold)
        if dData['light'].iloc[i-1] < 3 and dData['light'].iloc[i] > 3 and countedNoLight > 20:
            resetValues = True
            
        
        # Accumulate energy change
        
        deltaTime = dData['time'].iloc[i+1] - dData['time'].iloc[i]
        dT = dData['tempIn'].iloc[i] - initTemp - 2 

        dQ += dData['light'].iloc[i] * (1 - ALBEDO) * TRANSMISSION * A_EKSP * deltaTime
        
        dQ -= np.nan_to_num(h_air * tempDiff_air.iloc[i] * A_AIR * deltaTime)
        dQ -= np.nan_to_num(h_ground * tempDiff_ground.iloc[i] * A_FLOOR * deltaTime)




        # Calculate C, but only if the temperature difference is significant
        if abs(dT) > 1: C = dQ / dT
        else:           C = np.nan
        
        
        # Filter out extreme values so that the r2-filtering can work
        if C < -1e6 or C > 1e6: C = np.nan
        
        cList.append(C)
    
    
    return cList







# # %% MODELL for forventet lysmengde uavhengig av målte data

def modelInsolation(xTime, lat, lon):
    zInsolation = []

    for i, t in enumerate(xTime):
        day = dt.fromtimestamp(t).timetuple().tm_yday
        t -= 3600 * 2 # GMT+1 summer time
        
        # Solar angle theta
        theta = getSunAngleDeg(t, lat, lon, d=day)
    
        zInsolation.append(getInsolation(theta))
            
    
    return np.array(zInsolation)


#%% PLOTTING
def plotD(dData: pd.DataFrame, yInsolationModel: np.ndarray, saveFig: bool = True):
    """ Plot all data from the Drivhus """
    
    DPI = 600
    
    x = pd.to_datetime(dData['time'].astype(int), unit='s') 
    
    yTempIn = dData['tempIn']
    yTempOut = dData['tempOut']
    yTempGround = dData['tempGround']
    yBatVolt = dData['batVolt']
    yInsolation = dData['light']
    yInsolationModel = yInsolationModel
    


    # Battery Voltage
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Battery Voltage")

    ax.plot(x, yBatVolt, ".", label="Battery Voltage", color="navajowhite")
    ax.set_xlabel("Time")
    ax.set_ylabel("Battery Voltage [V]", color="black")
    # ax.set_ylim(2.5, 4.5)
    plt.title("Battery Voltage")
    ax.legend(loc='upper left')
    
    if saveFig: plt.savefig("plots/Battery Voltage.png", dpi=DPI, bbox_inches='tight')


    # Temp In
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temp In")

    ax.plot(x, yTempIn, "-", label="Inside Temperature", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    ax.set_ylim(-15, 45)
    plt.title("Temp In")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Temp In.png", dpi=DPI, bbox_inches='tight')


    # Temp In and Out
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temp In and Out")

    ax.plot(x, yTempIn, "-", label="Inside Temperature", color="red")
    ax.plot(x, yTempOut, "-", label="Outside Temperature", color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("Temp In and Out")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Temp In and Out.png", dpi=DPI, bbox_inches='tight')
    


    # Temperature Difference
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temperature Difference")

    temp_diff = yTempIn - yTempOut
    ax.plot(x, temp_diff, ".", label="Temperature Difference", color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature Difference [K]", color="black")
    # ax.set_ylim(-5, 25)
    plt.title("Temperature Difference")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Temperature Difference.png", dpi=DPI, bbox_inches='tight')


    # Modelled Temperature in Ground
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Modelled Temperature in Ground")

    ax.plot(x, yTempGround, ".", label="Modelled Ground Temp", color="purple")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("Modelled Temperature in Ground")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Modelled Temperature in Ground.png", dpi=DPI, bbox_inches='tight')


    # All Temperatures
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("All Temperatures")

    ax.plot(x, yTempOut, ".", label="Outside Temperature", color="blue")
    ax.plot(x, yTempIn, ".", label="Inside Temperature", color="red")
    ax.plot(x, yTempGround, ".", label="Ground Temperature", color="purple")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("All Temperatures")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/All Temperatures.png", dpi=DPI, bbox_inches='tight')


    # Measured Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Measured Light")

    ax.plot(x, yInsolation / LUX_CONVERSION, ".", label="Measured Light", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Illuminance [Lx]", color="black")
    # ax.set_ylim(0, 6500)
    plt.title("Measured Light")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Measured Light.png", dpi=DPI, bbox_inches='tight')


    # Adjusted Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Adjusted Light")

    ax.plot(x, yInsolation, ".", label="Adjusted Light", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Insolation [W/m²]", color="black")
    # ax.set_ylim(0, 1400)
    plt.title("Adjusted Light")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Adjusted Light.png", dpi=DPI, bbox_inches='tight')


    # Adjusted Light and Model
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Adjusted Light and Model")

    ax.plot(x, yInsolation, ".", label="Adjusted Light", color="orange")
    ax.plot(x, yInsolationModel, "-", label="Insolation Model", color="yellow")
    ax.set_xlabel("Time")
    ax.set_ylabel("Insolation [W/m²]", color="black")
    # ax.set_ylim(0, 1400)
    plt.title("Adjusted Light and Model")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Adjusted Light and Model.png", dpi=DPI, bbox_inches='tight')
    
    
    # Temps and Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temps and Light")

    ax.plot(x, yTempOut, ".", label="Outside Temp", color="blue")
    ax.plot(x, yTempIn, ".", label="Inside Temp", color="red")
    ax2 = ax.twinx()
    ax2.plot(x, yInsolation, ".", label="Light", color="orange")

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    ax2.set_ylabel("Insolation [W/m²]", color="black")
    # ax2.set_ylim(0, 1400)
    plt.title("Temps and Light")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    if saveFig: plt.savefig("plots/Temps and Light.png", dpi=DPI, bbox_inches='tight')
    

def plotC(cData: dict, saveFig: bool = True):
    DPI = 600
    
    
    x = list(pd.to_datetime(cData['x'].astype(int), unit='s'))
    
    cList = cData['cList']
    cListClean = cData['cListClean']
    c = cData['c']
    cSTD = cData['cSTD']

    # Raw C
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Raw C")

    ax.plot(x, cList, ".", label="Raw C", color="lightblue")
    ax.set_xlabel("Time")
    ax.set_ylabel("C [J/K]", color="black")
    ax.set_ylim(0, 1e6)
    plt.title("Raw C")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Raw C.png", dpi=DPI, bbox_inches='tight')

    # Clean C
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Calculated C and Standard Deviation")

    ax.plot(x, cListClean, ".", label="C", color="lightblue")
    ax.hlines(c, x[0], x[-1], colors='b', linestyles='-')
    ax.hlines([c + cSTD, c - cSTD], x[0], x[-1], colors='g', linestyles='--')
    ax.set_xlabel("Time")
    ax.set_ylabel("C [J/K]", color="black")
    ax.set_ylim(0, 1e6)
    plt.title("Calculated C and Standard Deviation")
    ax.legend(loc='upper left')

    if saveFig: plt.savefig("plots/Clean C.png", dpi=DPI, bbox_inches='tight')





# %%
if __name__ == "__main__":
    
    # Location of Drivhus at Skien VGS
    lat = 59.200
    lon = 9.612
    
    
    # Fetch data from Drivhus and model insolation
    dData = drivhusData(fileName="Blu1_DATALOG220425.TXT")
    yInsolationModel = modelInsolation(dData['time'], lat, lon)

    
    cList = []
    xCList = []
    
    
    parts = {
        "first": {
            "start": 0,
            "end": 10000,
  
            "h_air": 1.47241,
            "h_ground": 8.12069,
            "h_air_range": np.linspace(0, 6, 30),
            "h_ground_range": np.linspace(8, 15, 30),
        },
        "second": {
            "start": 10000,
            "end": 18000,
        
        # Funnet manuelt 2 og 6.
        # Dårlige resultater, små variasjoner i U hjelper ikke
            "h_air": 2,
            "h_ground": 6,
            "h_air_range": np.linspace(1, 3, 30),
            "h_ground_range": np.linspace(5, 7, 30),
        },
        "third": {
            "start": 18000,
            "end": len(dData),
            
            "h_air": 5.03448,
            "h_ground": 9.3448,
            "h_air_range": np.linspace(4, 6, 30),
            "h_ground_range": np.linspace(7, 11, 30),
        }
    }

    # Toggle to either search for h values or use the given ones
    if searchForH := False:
        for key, part in parts.items():
            start = part['start']
            end = part['end']
            
            xList = dData.iloc[start : end]['time']

            # Find the best h values and calculate C
            (h_air, h_ground), cListPart = find_best_h(
                dData.iloc[start : end],
                part['h_air_range'],
                part['h_ground_range']
            )
            
            # Save the results
            part["h_air"] = h_air
            part["h_ground"] = h_ground

            cList.extend(cListPart)
            xCList.extend(xList)
            
    
    else:
        for key, part in parts.items():
            start = part['start']
            end = part['end']
            
            xList = dData.iloc[start : end]['time']
            
            # Calculate C with the given h values
            cListPart = calculateCWithParams(
                dData.iloc[start : end],
                part['h_air'],
                part['h_ground']
            )
                        
            cList.extend(cListPart)
            xCList.extend(xList) 
        
    
    
    cListClean = cleanData(cList, lower=-1e5, upper=1e6, upperDerivative=5000)
    
    # Calculate median and std
    c = np.nanmedian(cListClean)
    cSTD = np.nanstd(cListClean)
    
    
    print(np.average([1.5, 2, 5]))
    print(np.average([8.1, 6, 9.3]))
    
    print(np.std([1.5, 2, 5]))
    print(np.std([8.1, 6, 9.3]))
    
    cData = {
        "x": np.array(xCList),
        "cList": cList,
        "cListClean": cListClean,
        "c": c,
        "cSTD": cSTD
    }


    # Print the results
    for key, part in parts.items():
        print(f"{key}:")
        print(f"  U_air: {part['h_air']}")
        print(f"  U_ground: {part['h_ground']}")
        print()
        
    print(f"  c: {c}")
    print(f"  cSTD: {cSTD}")
    
    
    
    #%% plot all
    
    plotD(dData, yInsolationModel, saveFig=False)
    plotC(cData, saveFig=False)
    
    plt.show()
    
    
    
    