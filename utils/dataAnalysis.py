import numpy as np
import pandas as pd
from datetime import datetime as delta_temp
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.settingsAndConstants import A_AIR, A_FLOOR, A_EKSP, ALBEDO, TRANSMISSION, range_steps, std_treshold, max_iterations
from utils.fileUtils import saveParamsToFile, loadParamsFromFile
from utils.settingsAndConstants import start_day, num_days, day_nums


def evaluateSTDFromParams(dData, u_plastic, u_concrete):
    c_list = calculateCWithParams(dData, u_plastic, u_concrete)
    if not c_list or np.all(np.isnan(c_list)): return None  # ugyldig
    
    c = np.nanmedian(c_list)
    std = np.nanstd(c_list, mean=c)
    if np.isnan(c) or np.isnan(std) or c <= 0: return None # ugyldig
    
    return std, u_plastic, u_concrete
    
    

def find_best_u(
    dataset,
    u_plastic_start=0.1,
    u_plastic_end=10,
    u_concrete_start=0.1,
    u_concrete_end=10,
    steps=30,
    std_threshold=100,
    max_iterations=5,
    verbose=True
):
    """Finn parameterne (u_plastic, u_concrete) som gir laveste std. Start grovt med det gitte området. Lag deretter nytt område rundelta_temp de beste verdiene for å søke med finere oppløsning. Gjenta frem til endringen i standardavviket fra en til den neste iterasjonen er under terskelen."""
    
    best_std = float("inf")
    best_params = (None, None)
    history = []

    # Finn nye, finere områder opp til max_iterations ganger
    for iteration in range(max_iterations):
        # Lag en range av verdier for u_plastic og u_concrete gitt start- og sluttargumentene
        u_plastic_values = np.linspace(u_plastic_start, u_plastic_end, steps)
        u_concrete_values = np.linspace(u_concrete_start, u_concrete_end, steps)

        # Generer en 1D-liste av (u_plast, u_betong)-tupler med alle kombinasjoner av u_plastic og u_concrete ut ifra hver sin range av verdier
        combos = [(u_plastic, u_concrete) for u_plastic in u_plastic_values for u_concrete in u_concrete_values]

        
        # results er en liste av tupler=(c, std, u_plastic, u_concrete) for hver kombinasjon av u_plastic og u_concrete 
        # Bruk multiprocessing for å evaluere C og STD for alle kombinasjoner parallelt
        results = []
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(evaluateSTDFromParams, dataset, up, uc) for up, uc in combos]
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if verbose: print(f"\rIterasjon {iteration + 1}: {i + 1} / {len(futures)}", end="", flush=True)
                    
                    # Filtrer ut alle resultater med nan eller None
                    if result is not None and all(not np.isnan(x) for x in result[:2]):
                        results.append(result)
                
                except Exception as e:
                    print(f"\nFeil: {e}")
            else:
                print()  # newline etter telleren som bruker end=""

        # Avbryt hvis ingen resultater ble funnet
        if not results:
            break

        # Finn kombinajsonen av u_plastic og u_concrete som ga lavest standardavvik (std)
        local_best = min(results, key=lambda r: r[0])  # r[0] = std
        std, u_plastic, u_concrete = local_best

        history.append((u_plastic, u_concrete, std))

        if verbose: print(f"Beste i iterasjon {iteration + 1}: std = {std:.2f} ved u_plastic = {u_plastic:.4f}, u_concrete = {u_concrete:.4f}")

        # Oppdater beste standardavvik og parametere hvis de nye er bedre
        if std < best_std:
            best_params = (u_plastic, u_concrete)
            if best_std - std < std_threshold:
                break # Avslutt søket hvis endringen i standardavviket er under terskelen
            best_std = std
            

        # Velg verdier i en range av en kvart av det originale området,
        # sentrert rundelta_temp de beste verdiene funnet så langt.
        # Zoom inn rundelta_temp det beste området
        plastic_span = (u_plastic_end - u_plastic_start) / 4
        concrete_span = (u_concrete_end - u_concrete_start) / 4
        u_plastic_start = max(u_plastic - plastic_span, 0)
        u_plastic_end = u_plastic + plastic_span
        u_concrete_start = max(u_concrete - concrete_span, 0)
        u_concrete_end = u_concrete + concrete_span


    return best_params




def calculateCWithParams(dData: pd.DataFrame, u_plastic: float, u_concrete: float):
    tempDiff_air = dData['tempIn'] - dData['tempOut']
    tempDiff_ground = dData['tempIn'] - dData['tempGround']
    
    c_list = []
    
    reset_values = True
    
    for i in range(0, len(dData)):
        
        if i+1 >= len(dData) or i-1 < 0:
            c_list.append(np.nan)
            continue
        
        if reset_values:
            delta_Q = 0
            initTemp = dData['tempIn'].iloc[i-1]
            reset_values = False
        
        
        # Find delta_time and deltaTemperature (delta_temp)
        delta_time = dData['time'].iloc[i+1] - dData['time'].iloc[i]
        delta_temp = dData['tempIn'].iloc[i] - initTemp - 2 

        # Solar irradiation increases the energy
        delta_Q += dData['light'].iloc[i] * (1 - ALBEDO) * TRANSMISSION * A_EKSP * delta_time
        

        # Energy flows from the system of highest temperature to the system of lowest temperature
        # Use np.nan_to_num to prevent one faulty value from setting the entire delta_Q to NaN
        delta_Q -= np.nan_to_num(u_plastic * tempDiff_air.iloc[i] * A_AIR * delta_time)        
        delta_Q -= np.nan_to_num(u_concrete * tempDiff_ground.iloc[i] * A_FLOOR * delta_time)


        # Calculate C, but only if the temperature difference is significant
        # This prevents division by zero or very small numbers
        if abs(delta_temp) > 1: C = delta_Q / delta_temp
        else:           C = np.nan
        
        
        # Filter out extreme values
        if C < -1e6 or C > 1e6:
            C = np.nan
        
        c_list.append(C)
    
    
    return c_list



def searchDaysForU(days: list):
    if len(day_nums) == 1:
        print(f"Searching for new U-values for day {day_nums[0]}")
    else:
        print(f"Searching for new U-values for days {", ".join(map(str, day_nums[:-1]))} and {day_nums[-1]}")
        
    days_data = []
    
    stored_u_values: dict = loadParamsFromFile("u_values.json")
    if not stored_u_values:
        print("Ingen lagrede U-verdier funnet.")
    
    
    for i, day in enumerate(days[start_day-1 : start_day + num_days - 1]):
        day_num = i + start_day
        
        
        # Skip this day if it should not be analyzed
        if day_num not in day_nums:
            print(f"Dag {i + start_day} skal ikke analyseres.")
            continue
        
        
        
        print(f"\nAnalyzing day {day_num}")
        
        # Finn start- og slutt for søk av U-verdier hvis dagen er lagret i filen
        if stored_u_values.keys().__contains__(str(day_num)):
            stored_day = stored_u_values[str(day_num)]
            plastic_start, plastic_end = stored_day['plastic_range']
            concrete_start, concrete_end = stored_day['concrete_range']
        else:
            ... # Uses default imported values from utils.settingsAndConstants
        
        

        # Finn de beste parameterne for denne dagen
        (u_plastic, u_concrete) = find_best_u(
            day,
            plastic_start, plastic_end,
            concrete_start, concrete_end,
            steps=range_steps, std_threshold=std_treshold,
            max_iterations=max_iterations, verbose=True
        )


        # Lagre verdiene i en json-fil fortløpende i tilfelle programmet krasjer etter noen dager
        saveParamsToFile(
            file_name = "u_values.json",
            content = {str(day_num): {
                "plastic_range": [plastic_start, plastic_end],
                "concrete_range": [concrete_start, concrete_end],
                "u_plastic": u_plastic,
                "u_concrete": u_concrete
            }}
        )
        
        
        c_list = calculateCWithParams(day, u_plastic, u_concrete)
        
        data = {
            "u_plastic": u_plastic,
            "u_concrete": u_concrete,
            "c_list": pd.Series(c_list),
            "time": day['time'],
            "day": day_num
        }

        # Lagre parameterne i et nytt element i daysParams 
        days_data.append(data)
    
    return days_data



def generateCListFromFile(days: list):
    if len(day_nums) == 1:
        print(f"Generating C-List with known U-values for day {day_nums[0]}\n")
    else:
        print(f"Generating C-List with known U-values for days {", ".join(map(str, day_nums[:-1]))} and {day_nums[-1]}\n")
        
    # Load the parameters from the stored values
    stored_u_values = loadParamsFromFile("u_values.json")
    
    days_data = []

    for i, day in enumerate(days[start_day-1 : start_day + num_days - 1]):
        day_num = i + start_day
        
        if day_num not in day_nums:
            print(f"Dag {i + start_day} skal ikke analyseres.")
            continue
        
        if str(day_num) not in stored_u_values.keys():
            print(f"Dag {day_num} ikke funnet i lagrede verdier.")
            continue
        
        stored_day = stored_u_values[str(day_num)]
        
        u_plastic = stored_day['u_plastic']
        u_concrete = stored_day['u_concrete']
        
        c_list = calculateCWithParams(day, u_plastic, u_concrete)
        
        # Save the constants for each day
        dayParams = {
            "u_plastic": u_plastic,
            "u_concrete": u_concrete,
            "c_list": pd.Series(c_list),
            "time": day['time'],
            "day": day_num
        }
        
        days_data.append(dayParams)
    
    return days_data



def utcTodayNum(timestamp: int):
    timestamp = int(timestamp) # Ensure the timetamp is a valid integer
    
    return delta_temp.fromtimestamp(int(timestamp)).timetuple().tm_yday


def splitIntoDays(dData: pd.DataFrame):
    """Del opp dataene i dager
    - Del opp tid fra kl. 12:00 til 12:00 neste dag
    - Finn laveste temperatur hver dag
    - Finn klokkeslettet for laveste temperatur hver dag
    - Del opp tid med start og slutt i klokkeslett for laveste temperaturer
    """
    
    current_day = utcTodayNum(dData["time"].iloc[0])
    lowest_temp = float("inf")
    
    
    lowest_temp_IDs = []
    lowest_tempID = 0
    
    for id, timestamp, tempIn, *_ in dData.itertuples():
        timestamp = int(timestamp) + 86400/2 # Add 12 hours to switch days at 12 o'clock
        day_num = utcTodayNum(timestamp)
        
        if day_num > current_day:
            lowest_temp_IDs.append(lowest_tempID)
            
            current_day = day_num
            lowest_temp = float("inf")
            
        if tempIn < lowest_temp:
            lowest_tempID = id
            lowest_temp = tempIn
            

    # Add the last day
    else:
        lowest_temp_IDs.append(lowest_tempID)
    
    
    
    # Split data into parts between the lowest temperatures between two consecutive days
    days: list[pd.DataFrame] = []
    for i in range(len(lowest_temp_IDs) - 1):
        day_part = dData.iloc[lowest_temp_IDs[i] : lowest_temp_IDs[i + 1]].copy()
        days.append(day_part)
    
    return days