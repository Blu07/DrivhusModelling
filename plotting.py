import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Plotting


def plotRawData(x, yTempOut, yTempIn, yInsolation):
    
    fig, ax1 = plt.subplots()
    fig.canvas.manager.set_window_title("Raw Data")  # Setter vindustittel
    
    ax1.plot(x, yTempOut, ".", label="Outside Temperature", color="blue")
    ax1.plot(x, yTempIn, ".", label="Inside Temperature", color="red")

    ax1.set_title("Temperature and Insolation Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature [°C]", color="black")
    ax1.set_ylim(-15, 30)

    ax2 = ax1.twinx()
    ax2.plot(x, yInsolation, ".", label="Insolation", color="orange")
    
    ax2.set_ylabel("Insolation [W/m^2]", color="black")
    ax2.set_ylim(0, min(1400, max(yInsolation) * 1.2))

    fig.legend(loc="upper left")



def plotRawInsolation(x, y):
    
    plt.figure("Raw Insolation")
    plt.plot(x, y, ".", label="Recorded Insolation", color="orange")

    plt.title("Insolation over time")
    plt.ylabel("Inoslation [W/m^2]")

    plt.ylim(0, min(1400, max(y)))
    plt.legend(loc="upper left")
 
    
def plotBatteryVoltage(x, y):
    
    plt.figure("Battery Voltage")
    plt.plot(x, y, ".", label="Battery Voltage", color="navajowhite")
    
    plt.title("Battery Voltage over time")
    plt.ylabel("Voltage [V]")
    plt.ylim(2.5, 4.5)
    plt.legend(loc="upper right")


def plotInterpolatedTemperatures(x, y, zOut, zIn):
    
    fig = plt.figure("Interpolated Temperatures")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, zOut, color="blue")
    ax.scatter(x, y, zIn, color="red")


    plt.title("Temperatures throughout every day")
    ax.set_xlabel(f'Day of Year [Julian Day]')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Temperature [°C]')


def plotInterpolatedInsolation(x, y, z):
    
    fig = plt.figure("Interpolated Insolation")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color="orange")


    plt.title("Light levels throughout every day")
    ax.set_xlabel(f'Day of Year [Julian Day]')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Light [W/m^2]')

    ax.set_zlim(0, min(1400, np.nanmax(z)))


def plotInsolationModel(x, y, z):

    fig = plt.figure("Insolation Model")
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(x, y, z)


    # Label the axes
    ax.set_xlabel(f'Day of Year')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Light [W/m^2]')

    ax.set_zlim(0, 1500)

    plt.title("Light Levels throughout every day")


def plotSolarAngleModel(x, y, z):
    
    fig = plt.figure("Solar Angle Model")
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(x, y, z)


    # Label the axes
    ax.set_xlabel(f'Day of Year')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Solar Angle [°]')

    ax.set_zlim(0, 90)

    plt.title("Solar Angle throughout every day")


def plotCloudCoverModel(x, y, z):
    
    fig = plt.figure("Cloud Cover Model")
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(x, y, z)


    # Label the axes
    ax.set_xlabel(f'Day of Year')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Cloud Cover [%]')

    ax.set_zlim(0, 100)

    plt.title("Cloud Cover Percent throughout every day")


def plotTempChangeModel(DF, model, timeStep):
    # Plot Temp Change Model
    
    fig = plt.figure("Temperature Change Model")
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(DF['temp_diff'], DF['light_in'], DF['temp_change'])


    # Lag en meshgrid for å tegne flaten
    x_surf = np.linspace(DF['temp_diff'].min(), DF['temp_diff'].max(), 20)
    y_surf = np.linspace(DF['light_in'].min(), DF['light_in'].max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    # Beregn predikerte verdier basert på modellen
    exog = pd.DataFrame({'const': 1, 'temp_diff': x_surf.ravel(), 'light_in': y_surf.ravel()})
    z_surf = model.predict(exog).values.reshape(x_surf.shape)

    # Plott flaten
    ax.plot_surface(x_surf, y_surf, z_surf, color='None', alpha=0.5)


    # Label the axes
    ax.set_xlabel('Temp. Difference [°C]')
    ax.set_ylabel('Insolation [W/m^2]')
    ax.set_zlabel(f'Temp. Change in {timeStep} min. [°C]')

    plt.title("Temperature Change based on Insolation and Temperature Difference")


def plotTemperatures(x, y, zOut = None, zIn = None):
    # Plot Temperature
    
    fig = plt.figure("Modelled and Simulated Temperatures")
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    if zOut is not None:
        ax.scatter(x, y, zOut)
    if zIn is not None:
        ax.scatter(x, y, zIn)

    # Label the axes
    ax.set_xlabel(f'Day of Year [Julian Day]')
    ax.set_ylabel('Time of Day [Hour]')
    ax.set_zlabel('Air Temperature [°C]')

    ax.set_zlim(-15, 60)

    plt.title("Temperature throughout every day")


