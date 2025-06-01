from datetime import datetime as dt
import numpy as np


def getSunAngleDeg(unix_time: int, lat, lon, d=None):
    date_time = dt.fromtimestamp(unix_time)
    
    # Local time in minutes
    LT = date_time.hour * 60 + date_time.minute

    # Local Standard Time Meridian (LSTM)
    LSTM = 15 * 1  # 15 degrees per hour for UTC+1

    # Day of the year
    if d is None:
        d = date_time.timetuple().tm_yday
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

def get_insolation(theta):
    insolation = 1361 * np.sin(theta)  # W/m^2, solar constant at top of atmosphere https://no.wikipedia.org/wiki/Solkonstanten
    
    if insolation < 0: return 0 # Negative insolation is not physically meaningful, so we set it to 0
    else: return insolation



def getFirstSunRiseTime(day, timeList, lat, lon):
    for t in timeList:
        sun_angle = getSunAngleDeg(t, lat, lon, day)

        if sun_angle > 0:
            return t




def modelInsolation(x_time, lat, lon):
    z_insolation = []

    for i, t in enumerate(x_time):
        day = dt.fromtimestamp(t).timetuple().tm_yday
        t -= 3600 * 2 # GMT+1 summer time
        
        # Solar angle theta
        theta = getSunAngleDeg(t, lat, lon, d=day)
    
        z_insolation.append(get_insolation(theta))
            
    
    return np.array(z_insolation)



