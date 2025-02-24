import requests
import numpy as np
from datetime import datetime as dt, timedelta
import json
import pandas as pd

def fetchDrivhusData(firstID=0):
    
    
    drivhusURL = 'https://drivhus.bluwo.me/get_data.php'
    data = {
        "fromID": firstID
    }
    
    response = requests.post(drivhusURL, json=data)
    statusCode = response.status_code
    
    match statusCode:
        case 200:
            data = response.json()
            
            if len(data) == 0:
                print("No data from Drivhus.")
                return []
            
            
            print(f"Received {len(data)} rows from Drivhus.")
            
            return data
        
        case _:
            print(f"Got status code {statusCode} and body {response.text} from Drivhus")
        
def cleanData(data, lower, upper, errors=[], calibration=0):
 
    cleansedData = []
    for value in data:
        invalid = \
            (value in errors) or \
            value < lower or \
            value > upper or \
            np.isnan(value)
        
        cleansedData.append(np.nan if invalid else value + calibration)

    return cleansedData




def openStoredData(filename) -> dict:
    with open(filename, 'r') as read_file:
        storedData: dict = json.load(read_file)
        return storedData
    
def saveUpdatedData(filename, data) -> None:
    with open(filename, 'w') as write_file:
        json.dump(data, write_file, indent=4)

def hasMETExpired(data: dict) -> bool:
    expiresDateTime = dt.strptime(data.get("Expires"), "%a, %d %b %Y %H:%M:%S %Z") + timedelta(hours=1) # GMT+1
    
    if dt.now() < expiresDateTime:
        return False
    
    return True


def fetchMETData(fileName, lat, lon):
    storedData: dict = openStoredData(fileName)

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
    firstTime = int(dt.fromisoformat(MET_data['properties']['timeseries'][0]['time']).timestamp())

    for time in existingTimes:
        if time >= firstTime:
            storedData['rows'].pop(str(time))
    
    
    
    # Insert the new values
    for hour in MET_data['properties']['timeseries']:
        time = int(dt.fromisoformat(hour['time']).timestamp())
        MET_row = hour['data']['instant']['details']
        
        row = {
            'temp': MET_row['air_temperature'],
            'cloud': MET_row['cloud_area_fraction']
        }
        
        storedData['rows'].update({str(time): row})
    
    
    saveUpdatedData(fileName, storedData)

    return storedData

def METData(fileName, lat, lon):
    data: dict = fetchMETData(fileName, lat, lon)['rows']
    
    xMET = []
    yMETTemp = []
    yMETCloud = []
    
    for time, values in data.items():
        xMET.append(int(dt.fromtimestamp(int(time)).timestamp()))
        
        yMETTemp.append(values['temp'])
        yMETCloud.append(values['cloud'])    
    
    return pd.Series(xMET), pd.Series(yMETTemp), pd.Series(yMETCloud)
  
