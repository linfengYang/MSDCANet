from Logger import log
import socket



params_appliance = {


    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'Min_on': 12,
        'Min_off': 0},
    'microwave': {
        'windowlength': 599, #249
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'Min_on': 12,
        'Min_off': 30},
    'fridge': {
        'windowlength': 600,#599
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'Min_on': 60,
        'Min_off': 12},
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'Min_on': 1800,
        'Min_off': 1800},
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'Min_on': 1800,
        'Min_off': 160}
    }


log('Parameters: ')
machine_name = socket.gethostname()
log('Machine name: ' + machine_name)
