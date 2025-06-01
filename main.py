# %% Import libraries

from utils.dataAnalysis import searchDaysForU, generateCListFromFile, splitIntoDays
from utils.getData import getDrivhusData
from utils.plotting import plotDrivhus, plotCPerDay, printParams, showPlots


# %% Main function to run the analysis and plotting
if __name__ == "__main__":
    
    # Fetch data from Drivhus and model insolation
    drivhus_data = getDrivhusData(fileName="drivhusFinal.txt")

    # Lag en liste som skiller dataene ved hver av de laveste temperaturene (som er på morgenene)
    # Hver "dag" starter da omtrent kl. 05:00 – 07:00
    days = splitIntoDays(drivhus_data)
    
    # Bytt mellom True/False for å velge å gjøre/ikke gjøre følgende
    
    # days_data: Liste med dataene (median, std) som skal plottes
    if False: days_data = searchDaysForU(days) # Søk etter nye U-verdier
    else:    days_data = generateCListFromFile(days) # Bruk lagrede verdier fra fil
    
    if True: printParams(days_data) # Print u_plastic og u_concrete for hver dag
    if True: plotDrivhus(drivhus_data, save_fig=False) # Plot drivhus data for the whole period
    if True: plotCPerDay(days_data, save_fig=False) # Plot resultatene for C hver dag
    if True: showPlots() # Vis plottene (eller velg å bare lagre dem med save_fig=True over)
    
    
    
    