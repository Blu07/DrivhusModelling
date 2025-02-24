

# TODO: plot MET data in either 2D or 3D. Only for convenience as of right now, ight be useful later.


# %% Samle datapunkter for lys, temp-diff og temp endring inne.

# Modell for endring av inne-temp ut ifra lys inn og temp-diff (parametre for 3D-kurve-plan)
# - 
# Modell for lokale lysforhold (parametre for lys inn for hvert tidspunkt)



# %% Modell for lys-inn i lokale fohold
# Lag en liste med lister. Hver liste er én dag.
# Hver dag inneholder en liste med lysforoldene for hver spesifikke tid. f.eks. hver time.
# Lag en xListe med unix-tiden for antall sekunder siden 00:00:00
# Finn unix-tiden for 00:00:00 fosr DEN DAGEN
# Trekk denne verdien fra alle elementene i lista.
# Interpoler lista med lys til å passe med xListen med unix-tider etter kl. 00:00:00.
# Legg den interpolerte lista til i lista for alle dager

# For å visualisere:
# Plot alle punktene.

# Lag en funksjon for en linje som passer gjennom lysforholdet for alle dagene på de spesifikke tidene.
# F.eks. en funksjon som passer for lysforholdet kl. 08:00.
# Siden solen går opp tidligere om sommeren, vil denne måten finne en fuksjon som passer for data som går gjennom alle tider i året.
# lagre parameterne for funksjonen i hvert klokkeslett.
# Gi dette til PROGRAM 2.

print("Importerer Biblioteker")

from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import statsmodels.api as sm

from plotting import *
from getData import * 





def clearPercent(day: int):
    
    A = 10.097
    d = 44.163
    
    clearPercent = A * np.sin(0.021*day - 2.311) + d

    return clearPercent


def getSunAngle(unixTime: int, lat, lon, d=None):
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

def getInsolationAt(theta, cloud, d=None):    
    Sc = 1366 # Solar constant is 1366 kW/m^2
  
    if theta < 0:
        return 0
        pass
    
    
    AM = 0
    if d is not None:
        AM = cloud/np.cos(theta)
 
    # Formel for insolasjon
    insolation = Sc * np.e**(-AM) * np.cos(np.pi/2 - theta)
    
    return insolation


def getFirstSunRiseTime(day, timeList):
    for t in timeList:
        sunAngle = getSunAngle(t, lat, lon, day)

        if sunAngle > 0:
            return t



#%% FETCH DRIVHUS DATA AND CLEAN FAULTY READINGS
def drivhusData():

    print("Henter data fra Drivhus")
    dData = pd.DataFrame(fetchDrivhusData(firstID=19989))
    timeCalibration = 3600


    dData['time'] = cleanData(
        pd.to_numeric(dData['time'].astype(int)),
        lower=1700000000,
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
        calibration=0
    )


    dData.set_index('id', inplace=True)
    
    # data = combineFromIntervals(data) # To be worked on, using all data for now
    
    
    return dData





# %% INTERPOLER RÅ DATA TIL DAGLIGE VERDIER

def interpolateDrivhusData(dData, dayTimeList):

    # Lag en liste med lister. Hver liste er én dag.
    # Hver dag inneholder en liste med lysforoldene for hver spesifikke tid. f.eks. hver time.

    print("Lager modell for lokale lysforhold.")

    daysDict = dict()
    timesDaysList = dict()

    tempInList = list()
    tempOutList = list()
    insolationList = list()


    # Lagre alle verdiene i tidslista ut ifra hvilken dag på året de er i.
    for index, unixTime in dData['time'].items():

        time = dt.fromtimestamp(unixTime)
        dayNum = time.timetuple().tm_yday
        
        twleveOCLock = dt(time.year, time.month, time.day, 0, 0, 0).timestamp()
        
        timeOfDay = dData['time'][index] - twleveOCLock
        timesDaysList.setdefault(dayNum, []).append(timeOfDay)
        
        tempIn = dData['tempIn'][index]
        tempOut = dData['tempOut'][index]
        insolation = dData['light'][index]
        
        daysDict.setdefault(dayNum, []).append([tempIn, tempOut, insolation])
        

    # Lag interpolerte lister for hver dag
    for key in daysDict.keys():


        interpTempIn = np.interp(dayTimeList, timesDaysList[key], [x[0] for x in daysDict[key]], np.nan, np.nan)
        interpTempOut = np.interp(dayTimeList, timesDaysList[key], [x[1] for x in daysDict[key]], np.nan, np.nan)
        interpInsolation = np.interp(dayTimeList, timesDaysList[key], [x[2] for x in daysDict[key]], np.nan, np.nan)
        
        
        insolationList.append(interpInsolation)
        tempInList.append(interpTempIn)
        tempOutList.append(interpTempOut)
        
        
        

    # Prepare data to plot

    x = np.array([[i]*len(dayTimeList) for i in timesDaysList.keys()]).flatten()
    y = np.array([dayTimeList]*len(timesDaysList)).flatten()/3600

    zTempIn = np.array(tempInList).flatten()
    zTempOut = np.array(tempOutList).flatten()
    zInsolation = np.array(insolationList).flatten()/120

    
    return x, y, zTempIn, zTempOut, zInsolation


#%% MODELL FOR LOKALE LYSFORHOLD

def fitLocalCloudCover():
    """TODO
    """
    #--- LYS: Bruk regresjon for å finne tilpassede linjer for hvert klokkelsett
    
    # empiricalToExpectedRatio = empiricalZ / expectedZ
    
    def fitFunc(x, a, b):
        # Lag en funksjon som passer med buen som lyset lager gjennom året. 
        return a*x + b
        

    # dayParams = []

    # for i, time in enumerate(timeList):
    #     lightsAtTime = []
        
    #     for day in empiricalDaysList:
    #         lightsAtTime.append(day[i])
        
    #     x_data = np.array([i for i in range(len(lightsAtTime))])
    #     y_data = lightsAtTime
        
        
        
    #     params, covariance = curve_fit(fitFunc, x_data, y_data)
    #     a, b = params
        
    #     dayParams.append(params)


    # print(dayParams)




    # Lag en funksjon for en linje som passer gjennom lysforholdet for alle dagene på de spesifikke tidene.
    # F.eks. en funksjon som passer for lysforholdet kl. 08:00.
    # Siden solen går opp tidligere om sommeren, vil denne måten finne en fuksjon som passer for data som går gjennom alle tider i året.
    # lagre parameterne for funksjonen i hvert klokkeslett.

 



# %% MODELL for forventet lysmengde uavhengig av målte data

def modelInsolation(xMET, yMETCloud, dayTimeList):

    # Lagre MET data for skydekke i dict for hver dag (key = #day)    
    timeDaysDict = dict() # Lagre også tidspunktene for hver avlesning. Slik kan listen interpoleres
    cloudCoverDaysDict = dict()

    # Lagre alle verdiene i tidslista ut ifra hvilken dag på året de er i.
    for key, unixTime in xMET.items():

        time = dt.fromtimestamp(unixTime)
        dayNum = time.timetuple().tm_yday
        
        twleveOCLock = dt(time.year, time.month, time.day, 0, 0, 0).timestamp()
        
        timeOfDay = unixTime - twleveOCLock
        timeDaysDict.setdefault(dayNum, []).append(timeOfDay)
        
        cloud = yMETCloud[key]
        cloudCoverDaysDict.setdefault(dayNum, []).append(cloud)
        

    # Lag interpolerte lister for hver dag
    for key in timeDaysDict.keys():
        
        interpolated = np.interp(dayTimeList, timeDaysDict[key], cloudCoverDaysDict[key])
        cloudCoverDaysDict[key] = interpolated
        


    zInsolation = []
    zCloud = []
    zSolarAngle = []

    for d in range(1, 366):
        for i, t in enumerate(dayTimeList):
            
            # Adds support for using expected cloud cover from MET in the model.
            # TODO The current implementation does not differentiate between years.
            # if d in cloudDaysList.keys():
            #     cloudPercent = cloudDaysList[d][i]/100
            
            # else:
            cloudPercent = 1 - clearPercent(d)/100
            theta = getSunAngle(t, lat, lon, d=d)
            insolation = getInsolationAt(theta, cloudPercent, d)
            
            zCloud      .append(cloudPercent * 100)
            zSolarAngle .append(theta*180/np.pi)
            zInsolation .append(insolation)
            
            

    return np.array(zCloud), np.array(zSolarAngle), np.array(zInsolation)




# %% Modell for gjennomsnittstemperatur gjennom året

def modelDayTemps(dayTimeList):

    def dayTempModel(x, high, low, highTime, lowTime):
                
        A = (high - low) / 2
        d = (high + low) / 2
        
        upPeriod = (highTime-lowTime) * 2
        downPeriod = (24 - highTime + lowTime) * 2

        
        if x < lowTime:
            sine = np.sin(2*np.pi/downPeriod * (x - lowTime - downPeriod/4))
        
        elif lowTime <= x <= highTime:
            sine = np.sin(2*np.pi/upPeriod * (x - lowTime - upPeriod/4))
        
        else:
            sine = np.sin(2*np.pi/downPeriod * (x - highTime + downPeriod/4))

        return A * sine + d
        

    # Månedlige høy- og lavtemperaturer for Skien
    highs = [-1.8, -0.9, 3.5, 9.1, 15.5, 20.4, 21.5, 20.1, 15.1, 9.3, 3.2, -0.5]
    lows =  [-6.8, -6.8, -3.3, 0.8, 5.5, 10.5, 12.2, 11.3, 7.5, 3.8, -1.5, -5.6]

    monthList = np.linspace(1, 12, 12, endpoint=True)/12 * 365
    yearList = np.linspace(1, 365, 365, endpoint=True)


    interpolatedLows = np.interp(yearList, monthList, lows)
    interpolatedHighs = np.interp(yearList, monthList, highs)



    z = []

    for d in range(0, 365):
                
        sunRiseTime = getFirstSunRiseTime(day=d, timeList=dayTimeList)
        sunRiseTime = sunRiseTime / 3600 + 0.5 # Convert from seconds to hours and add half an hour.
        
        
        for i, timeSec in enumerate(dayTimeList):
            
            # (?) Add support for adding registered temperature data to the data set: (example from cloud coverage)
            
            # if d in cloudDaysList.keys():
            #     cloudPercent = cloudDaysList[d][i]/100
            
            # else:
            #     cloudPercent = 1 - clearPercent(d)/100
            
            timeHour = timeSec / 3600
            temp = dayTempModel(timeHour, interpolatedHighs[d], interpolatedLows[d], highTime = 15, lowTime=sunRiseTime)
            
            z.append(temp)



    z = np.array(z)

    return z



# %% MODELL FOR ENDRING AV INNETEMPERATUR
def fitTempChange(dData, resolution=100, radius=5):

    print("Lager modell for endring av innetemperatur.")

    # Interpoler temp inne, temp ute og lysmengde
    # Dette gir et datasett med jevne mellomrom over alle verdier.

    start = dData['time'].iloc[0]
    stop = dData['time'].iloc[-1]
    
    interpolationTimeSet = np.linspace(start=start, stop=stop, num=resolution, dtype=int, endpoint=True)


    lightInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['light'])
    tempInInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['tempIn'])
    tempOutInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['tempOut'])


    tempDiff = tempInInterpolated - tempOutInterpolated
    tempChange = np.zeros(resolution)


    def linearFitFunc(x, a, b):
        return a*x + b


    for i in range(0, resolution):
        
        temps = tempInInterpolated[max(i-radius, 0) : min(i+radius+1, resolution-1)]
        xList = [x for x in interpolationTimeSet[max(i-radius, 0) : min(i+radius+1, resolution-1)]]
        params, _ = curve_fit(linearFitFunc, xList, temps)
        a, b = params
        
        tempChange[i] = a


    

    df = pd.DataFrame({
        "temp_diff": tempDiff,
        "light_in": lightInterpolated/120,      # 120lx = 1 W/m^2
        "temp_change": tempChange*timeStep*60   # Konverter til "temp-change per timeStep minutter"
    })
    
    
    # Definer X (uavhengige variabler) og legg til konstant
    X = df[["temp_diff", "light_in"]]
    X = sm.add_constant(X)  # Legger til en konstant for skjæringspunktet

    # Bygg og tilpass modellen
    model = sm.OLS(df["temp_change"], X).fit()

    return df, model





# %% SIMULERING
def simulateTemperature(dayTimeList, zTemps, zLight, tempChangeModel):
    
    numPerDay = len(dayTimeList)
    
    intercept, coef_temp_diff, coef_insolation = tempChangeModel.params
    
    z = []
    
    for d in range(0, 365):
        
        currentTemp = 0
         
        sunRiseTimeSec = getFirstSunRiseTime(day=d, timeList=dayTimeList)

        
        for i, timeSec in enumerate(dayTimeList):
            
            outsideTemp = zTemps[numPerDay*d + i]
            insolation = zLight[numPerDay*d + i]
            
            # Let the inside tmep follow the outside temp down until sunrise.
            # The temperatures should be approximately the same after a night with no light
            if timeSec < sunRiseTimeSec:                
                currentTemp = outsideTemp

            else:    
                temp_diff = currentTemp - zTemps[numPerDay*d + i]
                predicted_temp_change = intercept + coef_temp_diff * temp_diff + coef_insolation * insolation
                currentTemp += predicted_temp_change
                
            z.append(currentTemp)
        
        
    z = np.array(z)
    
    return z



# %%
if __name__ == "__main__":
    
    lat = 59.200
    lon = 9.612
    timeStep = 15 # minutes
    
    # Lag en xListe med unix-tiden for antall sekunder siden 00:00:00
    numPerDay = 24*60//timeStep
    dayTimeList = np.linspace(0, 86400, num=numPerDay, endpoint=False)

    # X og Y koordinater for plotting av et helt år med data
    xFullYear = np.array([[i]*numPerDay for i in range(1, 366)]).flatten()
    yFullYear = np.array([dayTimeList]*365).flatten()/3600


    # Fetch data from Drivhus and MET
    dData = drivhusData()
    xMET, _, yMETCloud = METData("sigMet.json", lat, lon)
    
    
    
    # Time as dateTime-objects for raw data
    xTimeRawDrivhus = pd.to_datetime(dData['time'].astype(int), unit='s') 
    
    # Create models of data
    tempChangeDF, tempChangeModel = fitTempChange(dData, resolution=10000, radius=10)
    zAirTemps = modelDayTemps(dayTimeList)
    zCloud, zSolarAngle, zInsolation = modelInsolation(xMET, yMETCloud, dayTimeList)
    
    # Simulate indoor temperature
    zDrivhusTemps = simulateTemperature(dayTimeList, zAirTemps, zInsolation, tempChangeModel)
    
    xRawInterp, yRawInterp, zRawInterpTempIn, zRawInterpTempOut, zRawInterpInsolation = interpolateDrivhusData(dData, dayTimeList)


    
    # Plot all the models and simulation temperature
    print("Plotter grafer")
    plotBatteryVoltage(xTimeRawDrivhus, dData['batVolt'])
    plotRawData(xTimeRawDrivhus, dData['tempOut'], dData['tempIn'], dData['light']/120)
    plotInterpolatedTemperatures(xRawInterp, yRawInterp, zRawInterpTempOut, zRawInterpTempIn)
    plotInterpolatedInsolation(xRawInterp, yRawInterp, zRawInterpInsolation)
    
    plotSolarAngleModel(xFullYear, yFullYear, zSolarAngle)
    plotCloudCoverModel(xFullYear, yFullYear, zCloud)
    plotInsolationModel(xFullYear, yFullYear, zInsolation)
    
    plotTempChangeModel(tempChangeDF, tempChangeModel, timeStep)
    
    plotTemperatures(xFullYear, yFullYear, zAirTemps, zDrivhusTemps)
    plt.show()
