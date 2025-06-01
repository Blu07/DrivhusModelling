
#%% Physical constants
A_AIR = 42 # m^2
A_EKSP = 15 # m^2
A_FLOOR = 15 # m^2

ALBEDO = 0.4
TRANSMISSION = 0.8

LUX_CONVERSION = 1/7.5 # W/m^2 per Lx

#%% Plotting settings
DPI = 600

#%% Location of Drivhus at Skien VGS
lat = 59.200
lon = 9.612

#%% Settings for the analysis

# The days to include in the analysis
start_day = 1
num_days = 100
LAST_DAY = 43 # There is no more data
skip_days = [1, 23, 30, 31, 32, 36, 41] # These just dont work, some are missing data
skip_days.extend([5, 10, 13, 17, 18, 19, 24, 27]) # These give 0 (which is not correct)

# Generate a list of day numbers given the above parameters
day_nums = [d for d in range(start_day, start_day + num_days) if d not in skip_days and d <= LAST_DAY]

if len(day_nums) == 0:
    print("No days to analyze. Please check the settings.")
    exit(1)

#%% Settings for the U value search

std_treshold = 500  # Threshold between two consecutive searches
max_iterations = 8  # Maximum number of iterations per day
range_steps = 40  # Number of steps in each range of u_plastic and u_concrete

# Default ranges to search for plastic and concrete that are used
# if there are no stored values for the day in the json file.
plastic_start = 1
plastic_end = 5
concrete_start = 1
concrete_end = 15