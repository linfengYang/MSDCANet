import pandas as pd

applianceName='fridge'

input_file = f'created_data/UK_DALE/{applianceName}/{applianceName}_training_.csv'
data = pd.read_csv(input_file)


total_power = data.iloc[:, 0]
appliance_power = data.iloc[:, 1]


AGG_MEAN = 522
AGG_STD = 814
APPLIANCE_MEAN = 200
APPLIANCE_STD = 400


total_power_unscaled = total_power * AGG_STD + AGG_MEAN
appliance_power_unscaled = appliance_power * APPLIANCE_STD + APPLIANCE_MEAN


# total_power_unscaled.to_csv('DN_UK_DALE/total_power_unscaled.csv', index=False, header=['aggregate'])
appliance_power_unscaled.to_csv(f'created_data/DN_UK_DALE/{applianceName}_training_.csv', index=False, header=['power'])


