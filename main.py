


# %% Modell for lys-inn i lokale fohold - NOTATER, IDEER
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



# %% Import libraries
print("Importerer Biblioteker")

from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pprint import pprint

import statsmodels.api as sm

from utils.plotting import *
from utils.getData import * 
from utils.sunCalculations import *




def structureByDay(dData: pd.DataFrame):
    
    daysDict = {}
    
    # Lagre alle verdiene i tidslista ut ifra hvilken dag på året de er i.
    for i, unixTime in dData['time'].items():

        time = dt.fromtimestamp(unixTime)
        tempIn = dData['tempIn'].iloc[i]
        tempOut = dData['tempOut'].iloc[i]
        insolation = dData['light'].iloc[i]
        
        
        twleveOCLock = dt(time.year, time.month, time.day, 0, 0, 0).timestamp()
        timeOfDay = unixTime - twleveOCLock
        
        dayNum = time.timetuple().tm_yday

        daysDict.setdefault(dayNum, {}).setdefault("time", []).append(timeOfDay)
        daysDict.setdefault(dayNum, {}).setdefault("tempIn", []).append(tempIn)
        daysDict.setdefault(dayNum, {}).setdefault("tempOut", []).append(tempOut)
        daysDict.setdefault(dayNum, {}).setdefault("light", []).append(insolation)

    return daysDict

def calculateH(dData: pd.DataFrame):
    
    tempDiff = dData['tempIn'] - dData['tempOut']
    
    
    # Calculate h for intervals of 1h through each day and plot
            
    
    A_omg = 16 # m^2
    A_eksp = 4 # m^2
    
    alpha = 0.3
    tau = 0.8
    

    hList = []
    
    anyHasValue = False
    
    for i in range(0, len(dData)):
        
        t = dData['time'].iloc[i]
        if i + 1 == len(dData):
            hList.append(np.nan)
            continue
            
    
        deltaTime = dData['time'].iloc[i+1] - dData['time'].iloc[i]
        dTdt = (tempDiff[i+1] - tempDiff[i]) / deltaTime
        
        I = dData['light'].iloc[i+1]
        tDiff = tempDiff.iloc[i]

        # Values where the temperature difference is low are not intereesting.
        # Cut out all values where either the tmp diff is too low or temp_change is high.
        # Constant temp is required
        
        if abs(dTdt) < 0.01 and dData['tempIn'].iloc[i] > 15:
            h = ( ( 1-alpha ) * tau * A_eksp * I ) / ( A_omg * tDiff )   
            anyHasValue = True         
        
        else:
            h = np.nan
        
  
        
        hList.append(h)
    
    if anyHasValue:
        h = np.nanmean(hList)
    else:
        h = np.nan
        
    return h, hList


from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import linregress
import numpy as np

def evaluate_params(dData, h_air, h_ground):
    cList = calculateCWithParams(dData, h_air, h_ground)
    c = np.nanmedian(cList)
    avg = np.nanmean(cList)
    
    # if c < 0 or avg < 0:
    #     return None

    c_array = np.array(cList)
    x = np.arange(len(c_array))
    valid = ~np.isnan(c_array)

    if np.sum(valid) / len(c_array) < 0.1:
        return None
    
    x_valid = x[valid]
    c_valid = c_array[valid]
    
    slope, intercept, r_value, p_value, std_err = linregress(x_valid, c_valid)
    r_squared = r_value ** 2

    return r_squared, h_air, h_ground, cList


def find_best_h(dData, h_air_range, h_ground_range):
    best_r2 = -1
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
            
            r_squared, h_air, h_ground, cList = result
            
            if r_squared > best_r2:
                best_r2 = r_squared
                best_params = (h_air, h_ground)
                best_cList = cList
    


    return best_params, best_cList


def calculateCWithParams(dData: pd.DataFrame, h_air: float, h_ground: float):
    tempDiff_air = dData['tempIn'] - dData['tempOut']
    tempDiff_ground = dData['tempIn'] - dData['tempGround'] # Ground temp approx
    
    E_in = 0
    E_out = 0
    
    initTemp = dData['tempIn'].iloc[0] 
    
    countedNoLight = 1
        
    A_air = 16 # m^2
    A_floor = 4 # m^2
    
    alpha = 0.3
    tau = 0.8
    
    cList = []
    
    resetValues = True 
    
    for i in range(0, len(dData)):
        
        if i+1 >= len(dData) or i-1 < 0:
            cList.append(np.nan)
            continue
        
        if resetValues:
            E_in = 0
            E_out = 0
            countedNoLight = 0
            initTemp = dData['tempIn'].iloc[i-1]
            
            resetValues = False
        
        
        
        # Reset values when the sun sets (with 3 W/m^2 threshold)
        if dData['light'].iloc[i] < 3:
            countedNoLight += 1
            
        # Reset values when the tmep starts sinking between 15:00 and 15:10
        # timeOfDay = dt.fromtimestamp(dData['time'].iloc[i])
        # if timeOfDay.hour == 15 and timeOfDay.minute > 0 and timeOfDay.minute < 10:
        #     resetValues = True
         
        
        if dData['light'].iloc[i-1] < 3 and dData['light'].iloc[i] > 3 and countedNoLight > 20:
            resetValues = True
            
        
        # Accumulate energy change
        
        deltaTime = dData['time'].iloc[i+1] - dData['time'].iloc[i]
        dT = dData['tempIn'].iloc[i] - initTemp - 2 

        E_in += dData['light'].iloc[i] * (1 - alpha) * tau * A_floor * deltaTime
        
        E_out += np.nan_to_num(h_air * tempDiff_air.iloc[i] * A_air * deltaTime)
        E_out += np.nan_to_num(h_ground * tempDiff_ground.iloc[i] * A_floor * deltaTime)

        dQ = E_in - E_out

        if abs(dT) > 1:
            # print(round(dQ, 2), round(dT, 2))
            C = dQ / dT
        else:
            C = np.nan
        
        if C < -1e6 or C > 1e6:
            C = np.nan
        
        # print(C)

        cList.append(C)
    
    
    return cList







# # %% INTERPOLER RÅ DATA TIL DAGLIGE VERDIER

# def interpolateDrivhusData(dData, timeStep):

#     # Lag en liste med lister. Hver liste er én dag.
#     # Hver dag inneholder en liste med lysforoldene for hver spesifikke tid. f.eks. hver time.

#     print("Lager modell for lokale lysforhold.")
    
    
#     # list of the unix-time for seconds 00:00:00 - 23:59:59 at the given resolution
#     numPerDay = 24*60//timeStep
#     dayTimeList = np.linspace(0, 86400, num=numPerDay, endpoint=False)
    

#     daysDict = dict()
#     timesDaysList = dict()

#     tempInList = list()
#     tempOutList = list()
#     insolationList = list()


#     # Lagre alle verdiene i tidslista ut ifra hvilken dag på året de er i.
#     for index, unixTime in dData['time'].items():

#         time = dt.fromtimestamp(unixTime)
#         dayNum = time.timetuple().tm_yday
        
#         twleveOCLock = dt(time.year, time.month, time.day, 3, 0, 0).timestamp() # two hours forward; GMT+1 and summer time
        
#         timeOfDay = dData['time'][index] - twleveOCLock
#         timesDaysList.setdefault(dayNum, []).append(timeOfDay)
        
#         tempIn = dData['tempIn'][index]
#         tempOut = dData['tempOut'][index]
#         insolation = dData['light'][index]
        
#         daysDict.setdefault(dayNum, []).append([tempIn, tempOut, insolation])
        
    
    
#     # Lag interpolerte lister for hver dag
#     for day in daysDict.keys():


#         interpTempIn = np.interp(dayTimeList, timesDaysList[day], [x[0] for x in daysDict[day]], np.nan, np.nan)
#         interpTempOut = np.interp(dayTimeList, timesDaysList[day], [x[1] for x in daysDict[day]], np.nan, np.nan)
#         interpInsolation = np.interp(dayTimeList, timesDaysList[day], [x[2] for x in daysDict[day]], np.nan, np.nan)
        
        
#         insolationList.append(interpInsolation)
#         tempInList.append(interpTempIn)
#         tempOutList.append(interpTempOut)
        
        

#     # Prepare data to plot

#     x = np.array([[i]*len(dayTimeList) for i in timesDaysList.keys()]).flatten()    
#     y = np.array([dayTimeList]*len(timesDaysList)).flatten()/3600


#     zTempIn = np.array(tempInList).flatten()
#     zTempOut = np.array(tempOutList).flatten()
#     zInsolation = np.array(insolationList).flatten()
    
    
    
    
#     return x, y, zTempIn, zTempOut, zInsolation




# #%% MODELL FOR LOKALE LYSFORHOLD

# def fitLocalCloudCover():
#     """TODO
#     """
#     #--- LYS: Bruk regresjon for å finne tilpassede linjer for hvert klokkelsett
    
#     # empiricalToExpectedRatio = empiricalZ / expectedZ
    
#     def fitFunc(x, a, b):
#         # Lag en funksjon som passer med buen som lyset lager gjennom året. 
#         return a*x + b
        

#     # dayParams = []

#     # for i, time in enumerate(timeList):
#     #     lightsAtTime = []
        
#     #     for day in empiricalDaysList:
#     #         lightsAtTime.append(day[i])
        
#     #     x_data = np.array([i for i in range(len(lightsAtTime))])
#     #     y_data = lightsAtTime
        
        
        
#     #     params, covariance = curve_fit(fitFunc, x_data, y_data)
#     #     a, b = params
        
#     #     dayParams.append(params)


#     # print(dayParams)




#     # Lag en funksjon for en linje som passer gjennom lysforholdet for alle dagene på de spesifikke tidene.
#     # F.eks. en funksjon som passer for lysforholdet kl. 08:00.
#     # Siden solen går opp tidligere om sommeren, vil denne måten finne en fuksjon som passer for data som går gjennom alle tider i året.
#     # lagre parameterne for funksjonen i hvert klokkeslett.

 


# # %% MODELL for forventet lysmengde uavhengig av målte data

def modelInsolation(xTime, lat, lon):

    zInsolation = []
    zSolarAngle = []

    for i, t in enumerate(xTime):
        day = dt.fromtimestamp(t).timetuple().tm_yday
        t -= 3600 * 2 # GMT+1 sumemr time
        
        theta = getSunAngleDeg(t, lat, lon, d=day)
        
        zSolarAngle.append(theta)
        zInsolation.append(getInsolationAt(theta, day))
            
    
    return np.array(zSolarAngle), np.array(zInsolation)




# # %% MODELL for gjennomsnittstemperatur gjennom året

# def modelDayTemps(dayTimeList, highs, lows, lat, lon):

#     def dayTempModel(x, high, low, highTime, lowTime):
                
#         A = (high - low) / 2
#         d = (high + low) / 2
        
#         upPeriod = (highTime-lowTime) * 2
#         downPeriod = (24 - highTime + lowTime) * 2

        
#         if x < lowTime:
#             sine = np.sin(2*np.pi/downPeriod * (x - lowTime - downPeriod/4))
        
#         elif lowTime <= x <= highTime:
#             sine = np.sin(2*np.pi/upPeriod * (x - lowTime - upPeriod/4))
        
#         else:
#             sine = np.sin(2*np.pi/downPeriod * (x - highTime + downPeriod/4))

#         return A * sine + d
        

#     monthList = np.linspace(1, 12, 12, endpoint=True)/12 * 365
#     yearList = np.linspace(1, 365, 365, endpoint=True)


#     interpolatedLows = np.interp(yearList, monthList, lows)
#     interpolatedHighs = np.interp(yearList, monthList, highs)



#     z = []

#     for d in range(0, 365):
                
#         sunRiseTime = getFirstSunRiseTime(day=d, timeList=dayTimeList, lat=lat, lon=lon)
#         sunRiseTime = sunRiseTime / 3600 + 0.5 # Convert from seconds to hours and add half an hour.
        
        
#         for i, timeSec in enumerate(dayTimeList):
            
#             # (?) Add support for adding registered temperature data to the data set: (example from cloud coverage)
            
#             # if d in cloudDaysList.keys():
#             #     cloudPercent = cloudDaysList[d][i]/100
            
#             # else:
#             #     cloudPercent = 1 - clearPercent(d)/100
            
#             timeHour = timeSec / 3600
#             temp = dayTempModel(timeHour, interpolatedHighs[d], interpolatedLows[d], highTime = 15, lowTime=sunRiseTime)
            
#             z.append(temp)



#     z = np.array(z)

#     return z




# # %% MODELL FOR ENDRING AV INNETEMPERATUR
# def fitTempChange(dData, resolution=100, radius=5):

#     print("Lager modell for endring av innetemperatur.")

#     # Interpoler temp inne, temp ute og lysmengde
#     # Dette gir et datasett med jevne mellomrom over alle verdier.

#     start = dData['time'].iloc[0]
#     stop = dData['time'].iloc[-1]
    
#     interpolationTimeSet = np.linspace(start=start, stop=stop, num=resolution, dtype=int, endpoint=True)


#     lightInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['light'])
#     tempInInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['tempIn'])
#     tempOutInterpolated = np.interp(interpolationTimeSet, dData['time'], dData['tempOut'])


#     tempDiff = tempInInterpolated - tempOutInterpolated
#     tempChange = np.zeros(resolution)


#     def linearFitFunc(x, a, b):
#         return a*x + b


#     for i in range(0, resolution):
        
#         temps = tempInInterpolated[max(i-radius, 0) : min(i+radius+1, resolution-1)]
#         xList = [x for x in interpolationTimeSet[max(i-radius, 0) : min(i+radius+1, resolution-1)]]
#         params, _ = curve_fit(linearFitFunc, xList, temps)
#         a, b = params
        
#         tempChange[i] = a


    

#     df = pd.DataFrame({
#         "temp_diff": tempDiff,
#         "light_in": lightInterpolated,
#         "temp_change": tempChange*timeStep*60   # Konverter til "temp-change per timeStep minutter"
#     })
    
    
#     # Definer X (uavhengige variabler) og legg til konstant
#     X = df[["temp_diff", "light_in"]]
#     X = sm.add_constant(X)  # Legger til en konstant for skjæringspunktet

#     # Bygg og tilpass modellen
#     model = sm.OLS(df["temp_change"], X).fit()

#     return df, model




# # %% SIMULERING
# def simulateTemperature(dayTimeList, zTemps, zLight, tempChangeModel, lat, lon):
    
#     numPerDay = len(dayTimeList)
    
#     intercept, coef_temp_diff, coef_insolation = tempChangeModel.params
    
#     z = []
    
#     for d in range(0, 365):
        
#         currentTemp = 0
         
#         sunRiseTimeSec = getFirstSunRiseTime(day=d, timeList=dayTimeList, lat=lat, lon=lon)

        
#         for i, timeSec in enumerate(dayTimeList):
            
#             outsideTemp = zTemps[numPerDay*d + i]
#             insolation = zLight[numPerDay*d + i]
            
#             # Let the inside temp follow the outside temp down until sunrise.
#             # The temperatures should be approximately the same after a night with no light
#             if timeSec < sunRiseTimeSec:                
#                 currentTemp = outsideTemp

#             else:    
#                 temp_diff = currentTemp - zTemps[numPerDay*d + i]
#                 predicted_temp_change = intercept + coef_temp_diff * temp_diff + coef_insolation * insolation
#                 currentTemp += predicted_temp_change
                
#             z.append(currentTemp)
        
        
#     z = np.array(z)
    
#     return z





#%%





# %%
if __name__ == "__main__":
    
    
    # Location of Drivhus at Skien VGS
    lat = 59.200
    lon = 9.612
    
    
    # Fetch data from Drivhus
    dData = drivhusData(fileName="DATALOG.TXT")
    
    ySunAngle, yInsolationModel = modelInsolation(dData['time'], lat, lon)

    dDataFirst = dData.iloc[:10000]
    dDataSecond = dData.iloc[10000:]
    
    cList = []
    fullCList = []

    # Toggle this to either search for h or use own values
    if searchForH := False:
            
        # Intervals for first half of data set. 
        h_air_range = np.linspace(1, 2, 25)
        h_ground_range = np.linspace(8, 12, 50)
                
        (h_air_first, h_ground_first), cListFirst = find_best_h(dDataFirst, h_air_range, h_ground_range)
        
        
        # Intervals for first half of data set.
        h_air_range = np.linspace(0, 10, 50)
        h_ground_range = np.linspace(0, 10, 50)
        
        (h_air_second, h_ground_second), cListSecond = find_best_h(dDataSecond, h_air_range, h_ground_range)

        
    else:
        # First 10 000 entries, then the rest ~8400
        
        h_air_first = 1.95
        h_ground_first = 8.1
        h_air_second = 0.0
        h_ground_second = 3.06
        
        # Add the first and second parts
        cListFirst = calculateCWithParams(dData[:10000], h_air_first, h_ground_first)
        cListSecond = calculateCWithParams(dData[10000:], h_air_second, h_ground_second)
    

    cListFirstClean = cleanData(cListFirst, lower=-100_000, upper=400_000, upperDerivative=5000)
    cListSecondClean = cleanData(cListSecond, lower=0, upper=300_000, upperDerivative=5000)
        

    # Median and Standard Deviation for each
    cFirst = np.nanmedian(cListFirstClean)
    cSTDFirst = np.nanstd(cListFirstClean)
    
    cSecond = np.nanmedian(cListSecondClean)
    cSTDSecond = np.nanstd(cListSecondClean)
    
    print("h_air_first: ", h_air_first)
    print("h_ground_first: ", h_ground_first)
    
    print("h_air_second: ", h_air_second)
    print("h_ground_second: ", h_ground_second)
    
    print(f"First Median: {cFirst}")
    print(f"First STD: {cSTDFirst}")
    print(f"Second Median: {cSTDSecond}")
    print(f"Second STD: {cSecond}")
    

    # Time as dateTime-objects for plotting raw data
    # xTimeRawDrivhus = dData['time']
    xTimeRawDrivhus = pd.to_datetime(dData['time'].astype(int), unit='s') 
    



    # Plot all the models and simulation temperature
    print("Plotter grafer")
    
    # Battery Voltage
    # plotRawData(xTimeRawDrivhus, yBatVolt=dData['batVolt'])
    
    # Temperatures
    # plotData("Temp Out", xTimeRawDrivhus, yTempOut=dData['tempOut'])
    # plotData("Temp In", xTimeRawDrivhus, yTempIn=dData['tempIn'])
    # plotData("Temp In and Out", xTimeRawDrivhus, yTempOut=dData['tempOut'], yTempIn=dData['tempIn'])
    # plotData("Temperature Difference", xTimeRawDrivhus, yTempDiff = dData['tempIn']-dData['tempOut'])
    # plotData("Modelled Temperature in Ground", xTimeRawDrivhus, yTempGround=dData['tempGround'])
    # plotData("All Temperatures", xTimeRawDrivhus, dData['tempOut'], dData['tempIn'], yTempGround=dData['tempGround'])
    
    # # Light
    # plotData("Measured Light", xTimeRawDrivhus, yInsolation=dData['light'])
    # plotData("Light vs. Model", xTimeRawDrivhus, yInsolation=dData['light'], yInsolationModel=yInsolationModel)
    
    # plotData("Temps and Light", xTimeRawDrivhus, dData['tempOut'], dData['tempIn'], dData['light'])
    



    # C First Raw and Clean
    label = f"Raw C, h_air = {round(h_air_first, 3)}, h_ground = {round(h_ground_first, 3)}" 
    plotData(label, xTimeRawDrivhus[:10000], yC=cListFirst, h_air=h_air_first, h_ground=h_ground_first)
    plotData("Temperatures and Raw C", xTimeRawDrivhus[:10000], dDataFirst['tempOut'], dDataFirst['tempIn'], yTempGround=dDataFirst['tempGround'], yC=cListFirst)
    
    label = f"Clean C, C = {round(cFirst)}, STD = {round(cSTDFirst)}"
    plotData(label, xTimeRawDrivhus[:10000], yC=cListFirstClean, cLine=cFirst, cSTD=cSTDFirst)
    
    # C Second Raw and Clean
    label = f"Raw C, h_air = {round(h_air_second, 3)}, h_ground = {round(h_ground_second, 3)}"
    plotData(label, xTimeRawDrivhus[10000:], yC=cListSecond, h_air=h_air_second, h_ground=h_ground_second)
    plotData("Temperatures and Raw C", xTimeRawDrivhus[10000:], dDataSecond['tempOut'], dDataSecond['tempIn'], yTempGround=dDataSecond['tempGround'], yC=cListSecond)
    
    label = f"Clean C, C = {round(cSecond)}, STD = {round(cSTDSecond)}"
    plotData(label, xTimeRawDrivhus[10000:], yC=cListSecondClean, cLine=cSecond, cSTD=cSTDSecond)

    
    
    
    # def ## OLD CODE -------
    
    
    
    # Time step
    # timeStep = 60 # minutes
    # rawInterpTimeStep = 5 # minutes
    
    # list of the unix-time for seconds 00:00:00 - 23:59:59 at the given resolution
    # numPerDay = 24*60//timeStep
    # dayTimeList = np.linspace(0, 86400, num=numPerDay, endpoint=False)

    # # X og Y koordinater for plotting av et helt år med data
    # xFullYear = np.array([[i]*numPerDay for i in range(1, 366)]).flatten()
    # yFullYear = np.array([dayTimeList]*365).flatten()/3600


    # startID = 19989 # ID for the first row in "drivhus"
    # startID = 0
    
    
    # xMET, _, yMETCloud = METData("sigMet.json", lat, lon)
    

    # dataByDay = structureByDay(dData)


    # Månedlige høy- og lavtemperaturer for Skien
    # TODO: hente inn bedre data og legge tilrette for tilsvarende funksjon
    # highs = [-1.8, -0.9, 3.5, 9.1, 15.5, 20.4, 21.5, 20.1, 15.1, 9.3, 3.2, -0.5]
    # lows =  [-6.8, -6.8, -3.3, 0.8, 5.5, 10.5, 12.2, 11.3, 7.5, 3.8, -1.5, -5.6]
    
    
    
    # plotInterpolatedTemperatures(xRawInterp, yRawInterp, zRawInterpTempOut, zRawInterpTempIn)
    # plotInterpolatedInsolation(xRawInterp, yRawInterp, zRawInterpInsolation)
    # plotInterpolatedTempDiff(xRawInterp, yRawInterp, zRawInterpTempIn - zRawInterpTempOut)
    
    # plotAllLight(xRawInterp, yRawInterp, zRawInterpInsolation, xFullYear, yFullYear, zInsolation)
    
    # plotSolarAngleModel(xFullYear, yFullYear, zSolarAngle)
    # plotCloudCoverModel(xFullYear, yFullYear, zCloud)
    # plotInsolationModel(xFullYear, yFullYear, zInsolation)
    
    # plotTempChangeModel(tempChangeDF, tempChangeModel, timeStep)
    
    # plotTemperatures(xFullYear, yFullYear, zAirTemps, zDrivhusTemps)
    
    # analyze(dData, dayTimeList, xMET, yMETCloud, lat, lon)
    # exit()
    
    # xRawInterp, yRawInterp, zRawInterpTempIn, zRawInterpTempOut, zRawInterpInsolation = interpolateDrivhusData(dData, rawInterpTimeStep)
    # zCloud, zSolarAngle, zInsolation = modelInsolation(xMET, yMETCloud, dayTimeList)
    
    
    # Create models of data
    # tempChangeDF, tempChangeModel = fitTempChange(dData, resolution=10000, radius=10)
    # zAirTemps = modelDayTemps(dayTimeList, highs, lows, lat, lon)
    
    # Simulate indoor temperature
    # zDrivhusTemps = simulateTemperature(dayTimeList, zAirTemps, zInsolation, tempChangeModel, lat, lon)
    
    
    plt.show()

