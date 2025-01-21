
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
from MET_Secrets import client_ID, secret_ID

import json

import requests

from pprint import pprint


def cleanTimeData(data):
    
    lower = 1700000000
    upper = 1900000000
    errors = []
    
    cleansedData = []
    for value in data:
        invalid = (value in errors) or value <= lower or upper <= value
        
        cleansedData.append(np.nan if invalid else value + 3600) # adjust for GMT+1 
    
    return cleansedData


def cleanBatteryData(data):
    
    lower = 2.0
    upper = 4.5
    errors = []
    
    cleansedData = []
    for value in data:
        invalid = (value in errors) or value <= lower or upper <= value
        
        cleansedData.append(np.nan if invalid else value)
       
    return cleansedData


def cleanTemperatureData(data):
    
    lower = -25
    upper = 40
    errors = [-127, 85, 0]
    
    cleansedData = []
    for value in data:
        invalid = (value in errors) or value <= lower or upper <= value
        
        cleansedData.append(np.nan if invalid else value)
    
    return cleansedData


def cleanLightData(data):

    lower = 0
    upper = 100_000
    errors = []
    

    
    cleansedData = []
    for value in data:
        invalid = \
            (value in errors) or \
            value < lower or \
            upper < value or \
            np.isnan(value)
        
        cleansedData.append(np.nan if invalid else value)

    return cleansedData




def openStoredData(filename) -> dict:
    with open(filename, 'r') as read_file:
        storedData: dict = json.load(read_file)
        return storedData
    
def saveUpdatedData(filename, data) -> None:
    with open(filename, 'w') as write_file:
        json.dump(data, write_file, indent=4)


def hasMETExpired(data: dict) -> bool:
    expiresDateTime = datetime.strptime(data.get("Expires"), "%a, %d %b %Y %H:%M:%S %Z") + timedelta(hours=1) # GMT+1
    
    if datetime.now() < expiresDateTime:
        return False
    
    return True


def fetchMETData(lat, lon):
    storeFilename = 'sigMet.json'

    storedData: dict = openStoredData(storeFilename)

    if not hasMETExpired(storedData):
        print("Weather data has NOT expired. Using stored data.")
        return storedData

    
    
    print("Weather data HAS expired. Fetching new data.")
    
    lastModifiedTime = storedData.get("last-modified")
    MET_URL = f'https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}&altitude=4' # Langesund Coordinates
    headers = {
        'User-Agent': 'Blus Drivhus 1.0',
        'From': 'bluwo.me',
        'If-Modified-Since': lastModifiedTime
    }
    
    response = requests.get(MET_URL, headers=headers)
    statusCode = response.status_code
    
    # Handle and early return failed requests
    if statusCode != 200:
        if statusCode == 304:
            print("Weather data has not been updated since last fetch. Using previously downloaded data.")
            return storedData
        else:
            print(f"Got status code {statusCode} and body '{response.text}'")
            return None
        

    MET_data = response.json()
    
    # Update the Expires and last-modified fields
    storedData['Expires'] = response.headers.get("Expires")
    storedData['last-modified'] = response.headers.get("last-modified")
    
    
    # This removes the values that is replaced by the incoming values 
    existingTimes = list(map(lambda x: int(x), storedData['rows'].keys()))
    firstTime = int(datetime.fromisoformat(MET_data['properties']['timeseries'][0]['time']).timestamp())

    for time in existingTimes:
        if time >= firstTime:
            storedData['rows'].pop(str(time))
    
    
    
    # Insert the new values
    for hour in MET_data['properties']['timeseries']:
        time = int(datetime.fromisoformat(hour['time']).timestamp())
        MET_row = hour['data']['instant']['details']
        
        row = {
            'temp': MET_row['air_temperature'],
            'cloud': MET_row['cloud_area_fraction']
        }
        
        storedData['rows'].update({str(time): row})
    
    
    saveUpdatedData(storeFilename, storedData)

    return storedData

        

  
def METData(lat, lon):
    data: dict = fetchMETData(lat, lon)['rows']
    
    xMET = []
    yMETTemp = []
    yMETCloud = []
    
    for time, values in data.items():
        xMET.append(int(datetime.fromtimestamp(int(time)).timestamp()))
        
        yMETTemp.append(values['temp'])
        yMETCloud.append(values['cloud'])    
    
    return pd.Series(xMET), pd.Series(yMETTemp), pd.Series(yMETCloud)
  




def getSunAngle(unixTime: int, lat, lon):
    dateTime = datetime.fromtimestamp(unixTime)
    
    # Local time in minutes
    LT = dateTime.hour * 60 + dateTime.minute

    # Local Standard Time Meridian (LSTM)
    LSTM = 15 * 1  # 15 degrees per hour for UTC+1

    # Day of the year
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
    a = np.arcsin(np.sin(np.radians(lat)) * np.sin(decl) +
                  np.cos(np.radians(lat)) * np.cos(decl) * np.cos(np.radians(HRA)))

    
    
    return np.degrees(a)



def getInsolation(xList, xCloud, lat, lon):
    
    Sc = 1366 # Solar constant is 1366 kW/m^2

    yResult = np.zeros(len(xList))
    
    print(xList, xCloud, lon, lat)
    
    for i, unixTime in enumerate(xList):

        
        theta = np.radians(getSunAngle(unixTime, lat, lon))


        if theta < 0:
            yResult[i] = 0
            continue
        
        
        AM = 1 / np.cos(theta)
        
        if xCloud[i] is not np.nan:
            AM *= xCloud[i]/100 # Apply cloud converage if valid predictions
        else:
            AM *= 0

        # Formel for insolasjon
        insolation = Sc * np.e**(-AM) * np.cos(np.pi/2 - theta)
        
        yResult[i] = insolation

    return pd.Series(yResult)



def combineFromIntervals(data):
    # Combine every 5-min interval into one reading.
    previousTime = data['time'].iloc[0]

    sumTempIn = 0
    numTempIn = 0

    sumTempOut = 0
    numTempOut = 0
    
    sumLight = 0
    numLight = 0

    sumBat = 0
    numBat = 0

    clearResolutionSeconds = 200
    rowsToDrop = []
    for id, time, tempIn, tempOut, batVolt, light in data.itertuples():

        cutAVG = True if \
            time - previousTime < clearResolutionSeconds or \
            (numBat > 0 and abs(batVolt - sumBat/numBat) > 0.05) or \
            (numLight > 0 and abs(light - sumLight/numLight) > 10) or \
            (numTempIn > 0 and abs(tempIn - sumTempIn/numTempIn) > 0.5) or \
            (numTempOut > 0 and abs(tempOut - sumTempOut/numTempOut) > 0.5) \
            else False
        
        if cutAVG:
            # print(time, temp, batVolt)
            
            sumTempIn += tempIn
            numTempIn += 1
            
            sumTempOut += tempOut
            numTempOut += 1
            
            sumLight += light
            numLight += 1
            
            sumBat += batVolt
            numBat += 1
    
            rowsToDrop.append(id)
            
        else:
            
            if numTempIn != 0:     
                avgTempIn = round(sumTempIn / numTempIn, 2)
            else:
                avgTempIn = np.nan
            
            if numTempOut != 0:
                avgTempOut = round(sumTempOut / numTempOut, 2)
            else:
                avgTempOut = np.nan
                
                
            if numBat != 0:
                avgBat = round(sumBat / numBat, 2)
            else:
                avgBat = np.nan
                
                
            if numLight != 0:
                avgLight = round(sumLight / numLight, 2)
            else:
                avgLight = np.nan
                
                
            data.loc[id] = [previousTime, avgTempIn, avgTempOut, avgBat, avgLight]
                
            previousTime = time
        
            sumTempIn = tempIn
            numTempIn = 1
        
            sumTempOut = tempOut
            numTempOut = 1
             
            sumLight += light
            numLight += 1

            sumBat = batVolt
            numBat = 1
        

    newData = data.drop(index=rowsToDrop, inplace=False)
    
    return newData



def interpolate(xTarget: pd.Series, xList: pd.Series, yList: pd.Series):
    yResult = np.zeros(len(xTarget))
    
    currentXIndex = 0
    

    
    for i, x in enumerate(xTarget):
        currentXIndex = max(0, currentXIndex - 1)
        

       
        while currentXIndex < len(xList)-1:
            if xList.iloc[currentXIndex] > x:
                break
            
            currentXIndex += 1
        
        

        # Interpolerer mellom "dette punktet" og "neste punkt" i forhold til currentXIndex
        x0, y0 = xList.iloc[currentXIndex-1], yList.iloc[currentXIndex-1]
        x1, y1 = xList.iloc[currentXIndex], yList.iloc[currentXIndex]
        
        # Formel for interpolering mellom to punkter
        y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
        
        yResult[i] = y
    
    return pd.Series(yResult)




def tempFitFunc(x, A, d):
    global c, phi
    
    # return A * np.sin(c*x + phi) + d
    return A * np.sin(c*x + phi) + d




def fitTemperature(data):
    x0 = data['time'].iloc[0]

    startTime = datetime.fromtimestamp(x0)
    timeSince06 = startTime - datetime.replace(startTime, hour=6, minute=0, second=0)
    seconds = timeSince06.seconds

    global c, phi
    c = (2*np.pi) / (86400)
    phi = -x0 * c - seconds
    
    
    xData = data['time']
    yData = data['temp']

    popt, pcov = curve_fit(tempFitFunc, xData, yData)

    # into the future by 1 month
    futureTime = data['time'].iloc[-1] + 2628000 
    
    xModel = np.linspace(min(xData), futureTime, 1000)
    yModel = tempFitFunc(xModel, *popt)
    
    return xModel, yModel




def getStoredDrivhusData():
    with open("drivhus.json", 'r') as read_file:
        storedData = json.load(read_file)
        return storedData


def fetchDrivhusData():
    storedData = getStoredDrivhusData()
    if len(storedData) > 0:
        lastStoredID = storedData[-1]['id']
    else:
        lastStoredID = 0
    
    drivhusURL = 'https://drivhus.bluwo.me/get_data.php'
    data = {
        "fromID": lastStoredID
    }
    
    response = requests.post(drivhusURL, json=data)
    statusCode = response.status_code
    
    match statusCode:
        case 200:
            data = response.json()
            
            if len(data) == 0:
                print("No new data from Drivhus.")
                return storedData
            
            
            print(f"Received {len(data)} new rows from Drivhus.")
            storedData.extend(data)
            
            with open("drivhus.json", 'w') as write_file:
                json.dump(storedData, write_file, indent=4)
        
            return storedData
        
        case _:
            print(f"Got status code {statusCode} and body {response.text}")
        

    



if __name__ == '__main__':

    # Location details for Langesund
    # lat = 59.017
    # lon = 9.741
    
    # Location details for Skien VGS
    lat = 59.200
    lon = 9.612
    
    print("Henter data fra Drivhus")
    data = pd.DataFrame(fetchDrivhusData())
    
    data['time'] = cleanTimeData(pd.to_numeric(data['time'].astype(int))) #, unit='s'))
    data['tempIn'] = cleanTemperatureData(pd.to_numeric(data['tempIn'].astype(float)))
    data['tempOut'] = cleanTemperatureData(pd.to_numeric(data['tempOut'].astype(float)))
    data['batVolt'] = cleanBatteryData(pd.to_numeric(data['batVolt'].astype(float)))
    data['light'] = cleanLightData(pd.to_numeric(data['light']))

    data.set_index('id', inplace=True)

    # data = combineFromIntervals(data)

    c = 0
    phi = 0






    
    # Plotting

    
        # Generate the model
        # xModel, yModel = fitTemperature(data)
    xMET, yMETTemp, yMETCloud = METData(lat, lon)
    
    
    
    xSun = np.linspace(min(data['time']), max(xMET), 50000, dtype=np.int32) # +31536000 for one year ahead
    
    ySunAngle = pd.Series([getSunAngle(x, lat, lon) for x in xSun])
    
    
    
    yMETCloudInterpolated = interpolate(xSun, xMET, yMETCloud)
    
    
    
    ySunInsolation = getInsolation(xSun, yMETCloudInterpolated, lat, lon)
    
    
    yRealInsolation = interpolate(xSun, data['time'], data['light']) / 120
    
    



    
    # Convert from unixTime to DateTime Objects
    data['time'] = pd.to_datetime(data['time'].astype(int), unit='s') 
        # xModel = np.array([datetime.fromtimestamp(x) for x in xModel])
    xMET = np.array([datetime.fromtimestamp(x) for x in xMET])
    xSun = np.array([datetime.fromtimestamp(x) for x in xSun])
    

    
    
    
    
    
    # Plotting
    
    
    
    
    # Temperature
    fig, ax1 = plt.subplots()

    ax1.plot(data['time'], data['tempIn'], ".", label="Recorded Inside Temperature", color="red")
    ax1.plot(data['time'], data['tempOut'], ".", label="Recorded Outside Temperature", color="blue")
        # ax1.plot(xModel, yModel, label="Temp. Model", color="red")

    ax2 = ax1.twinx()
    
    ax2.plot(data['time'], data['tempIn']-data['tempOut'], ".-", label = "Temperature Difference")
    ax2.legend(loc= "upper right")
    
    ax1.set_ylabel("Temperature")
    ax1.set_ylim(-15, 30)
    ax1.legend(loc="upper left")
    
    

    # Customization
    plt.title("Temperature Over Time")
    plt.legend()
    
    plt.figure(2)
    plt.plot(xMET, yMETTemp, ".-", color="dodgerblue")
    plt.title("Forecasted Temperature Outside")
    
    
    
    # Figur 1: Solvinkel og skydekning
    plt.figure(3)

    plt.plot(xSun, ySunAngle, "-", label="Sun Angle", color="orange")
    plt.plot(xMET, yMETCloud, ".-", label="Forecasted Cloud Coverage", color="pink")
    plt.plot(xSun, yMETCloudInterpolated, ".-", label="Forecasted Cloud Coverage", color="lightpink")
    plt.ylabel("Sun Angle (°) / Cloud Coverage (%)")
    plt.ylim(0, 100)

    # Legg til legende
    plt.legend(loc="upper right")

    # Legg til tittel og etiketter
    plt.title("Sun Angle and Cloud Coverage")
    plt.xlabel("Time")

    # Figur 2: Insolasjon
    plt.figure(4)

    print(max(ySunInsolation))
    
    plt.plot(xSun, ySunInsolation, ".-", label="Excpected Insolation", color="yellow")
    plt.plot(xSun, yRealInsolation, ".-", label="Measured Insolation", color="red")
    plt.ylabel("Insolation (W/m²)")
    plt.ylim(0, 1366)

    # Legg til legende
    plt.legend(loc="upper right")

    # Legg til tittel og etiketter
    plt.title("Insolation")
    plt.xlabel("Time")


    
    # New Plot for Battery
    
    plt.figure(5)
    plt.plot(data['time'], data['batVolt'], ".", label="Battery Voltage", color="navajowhite")
    plt.ylim(1.5, 5)
    plt.legend(loc="upper right")
    
    plt.show()



