
import numpy as np
from datetime import datetime as dt

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
    Sc = 1366 # Solar constant is 1366 W/m^2
  
    if theta < 0:
        return 0
        pass
    
    
    AM = 0
    if d is not None:
        AM = cloud/np.cos(theta)
 
    # Formel for insolasjon
    insolation = Sc * np.e**(-AM) * np.cos(np.pi/2 - theta)
    
    return insolation


def getFirstSunRiseTime(day, timeList, lat, lon):
    for t in timeList:
        sunAngle = getSunAngle(t, lat, lon, day)

        if sunAngle > 0:
            return t
