import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from utils.settingsAndConstants import LUX_CONVERSION, lat, lon, DPI
from utils.sunCalculations import modelInsolation



def plotDrivhus(dData: pd.DataFrame, save_fig: bool = True):
    """ Plot all data from the Drivhus """
    
    x = pd.to_datetime(dData['time'].astype(int), unit='s') 
    
    y_temp_in = dData['tempIn']
    y_temp_out = dData['tempOut']
    y_temp_ground = dData['tempGround']
    y_bat_volt = dData['batVolt']
    y_insolation = dData['light']
    y_insolation_model = modelInsolation(dData['time'], lat, lon)
    


    # Battery Voltage
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Battery Voltage")

    ax.plot(x, y_bat_volt, ".", label="Battery Voltage", color="navajowhite")
    ax.set_xlabel("Time")
    ax.set_ylabel("Battery Voltage [V]", color="black")
    # ax.set_ylim(2.5, 4.5)
    plt.title("Battery Voltage")
    ax.legend(loc='upper left')
    
    if save_fig: plt.savefig("plots/Battery Voltage.png", dpi=DPI, bbox_inches='tight')


    # Temp In
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temp In")

    ax.plot(x, y_temp_in, "-", label="Inside Temperature", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    ax.set_ylim(-15, 45)
    plt.title("Temp In")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Temp In.png", dpi=DPI, bbox_inches='tight')


    # Temp In and Out
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temp In and Out")

    ax.plot(x, y_temp_in, "-", label="Inside Temperature", color="red")
    ax.plot(x, y_temp_out, "-", label="Outside Temperature", color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("Temp In and Out")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Temp In and Out.png", dpi=DPI, bbox_inches='tight')
    


    # Temperature Difference
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temperature Difference")

    temp_diff = y_temp_in - y_temp_out
    ax.plot(x, temp_diff, ".", label="Temperature Difference", color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature Difference [K]", color="black")
    # ax.set_ylim(-5, 25)
    plt.title("Temperature Difference")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Temperature Difference.png", dpi=DPI, bbox_inches='tight')


    # Modelled Temperature in Ground
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Modelled Temperature in Ground")

    ax.plot(x, y_temp_ground, ".", label="Modelled Ground Temp", color="purple")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("Modelled Temperature in Ground")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Modelled Temperature in Ground.png", dpi=DPI, bbox_inches='tight')


    # All Temperatures
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("All Temperatures")

    ax.plot(x, y_temp_out, ".", label="Outside Temperature", color="blue")
    ax.plot(x, y_temp_in, ".", label="Inside Temperature", color="red")
    ax.plot(x, y_temp_ground, ".", label="Ground Temperature", color="purple")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    plt.title("All Temperatures")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/All Temperatures.png", dpi=DPI, bbox_inches='tight')


    # Measured Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Measured Light")

    ax.plot(x, y_insolation / LUX_CONVERSION, ".", label="Measured Light", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Illuminance [Lx]", color="black")
    # ax.set_ylim(0, 6500)
    plt.title("Measured Light")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Measured Light.png", dpi=DPI, bbox_inches='tight')


    # Adjusted Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Adjusted Light")

    ax.plot(x, y_insolation, ".", label="Adjusted Light", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Insolation [W/m²]", color="black")
    # ax.set_ylim(0, 1400)
    plt.title("Adjusted Light")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Adjusted Light.png", dpi=DPI, bbox_inches='tight')


    # Adjusted Light and Model
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Adjusted Light and Model")

    ax.plot(x, y_insolation, ".", label="Adjusted Light", color="orange")
    ax.plot(x, y_insolation_model, "-", label="Insolation Model", color="yellow")
    ax.set_xlabel("Time")
    ax.set_ylabel("Insolation [W/m²]", color="black")
    # ax.set_ylim(0, 1400)
    plt.title("Adjusted Light and Model")
    ax.legend(loc='upper left')

    if save_fig: plt.savefig("plots/Adjusted Light and Model.png", dpi=DPI, bbox_inches='tight')
    
    
    # Temps and Light
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Temps and Light")

    ax.plot(x, y_temp_out, ".", label="Outside Temp", color="blue")
    ax.plot(x, y_temp_in, ".", label="Inside Temp", color="red")
    ax2 = ax.twinx()
    ax2.plot(x, y_insolation, ".", label="Light", color="orange")

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]", color="black")
    # ax.set_ylim(-15, 45)
    ax2.set_ylabel("Insolation [W/m²]", color="black")
    # ax2.set_ylim(0, 1400)
    plt.title("Temps and Light")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    if save_fig: plt.savefig("plots/Temps and Light.png", dpi=DPI, bbox_inches='tight')
    

def plotCPerDay(days_params: dict, save_fig: bool = True):
    
    plt.figure("C and STD every day", figsize=(100, 20))
    
    median_list = []
    
    # Iterate through each day's parameters and plot the C values and standard deviation
    for i, day in enumerate(days_params):
        
        dayNum = day['day']
        
        c_list = day['c_list']
        # If the list is empty or contains only NaN values, skip this day
        if c_list.empty or c_list.dropna().empty:
            print(f"Hopper over dag {dayNum} (ingen gyldige C-verdier)")
            continue
        
        x_c_list = day['time'].to_numpy()
        time = np.array([dt.fromtimestamp(v) for v in x_c_list])
        
        # cMean = np.nanmean(c_list)
        cMedian = np.nanmedian(c_list) # Median is more robust to outliers than mean
        cSTD = np.nanstd(c_list, mean=cMedian)
        
        median_list.append(cMedian)

        
        plt.plot(time, c_list, ".", label=f"Day {dayNum}")
        # plt.hlines(cMean, time[0], time[-1], colors='black', linestyles='-')
        plt.hlines(cMedian, time[0], time[-1], colors='b', linestyles='-')
        plt.hlines([cMedian + cSTD, cMedian - cSTD], time[0], time[-1], colors='g', linestyles='--')
        plt.ylim(-0.25e6, 1e6)
        plt.legend()
        
    
    # Calculate the overall median and standard deviation of the C values
    C = np.nanmedian(median_list)
    CSTD = np.nanstd(median_list, mean=C)
    
    print(f"Overall median C: {C:.2f} J/K")
    print(f"Overall STD of C: {CSTD:.2f} J/K")
        
    start_time = dt.fromtimestamp(int(days_params[0]["time"].iloc[0]))
    end_time = dt.fromtimestamp((days_params[-1]["time"].iloc[-1]))
    
    plt.hlines(y=C, xmin=start_time, xmax=end_time, colors='black', linestyles='-', label=f"Median C")
    plt.hlines(y=[C + CSTD, C - CSTD], xmin=start_time, xmax=end_time, colors='red', linestyles='--', label=f"STD of C")
    
    plt.xlim(start_time - pd.Timedelta(days=1), end_time + pd.Timedelta(days=2))
    plt.title("C values and STD for each day")
    plt.xlabel("Time")
    plt.ylabel("C [J/K]")
    plt.legend(loc='right')

    plt.grid()

    
    if save_fig: plt.savefig("plots/C values for each day.png", dpi=100, bbox_inches='tight')


def printParams(days_data):
    """ Print the parameters for each day in a readable format """
    for i, day in enumerate(days_data):
        print(f"Day {day['day']}:")
        print(f"  u_plastic: {round(day['u_plastic'], 3)}")
        print(f"  u_concrete: {round(day['u_concrete'], 3)}")
        print()
        
        
def showPlots():
    """ Show all plots in the current figure """
    plt.show()
