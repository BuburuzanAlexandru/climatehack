from datetime import datetime, time, timedelta
from random import randrange, choice
import os
import numpy as np
import xarray as xr
from config import config
from multiprocessing import Pool

dataset = xr.open_dataset(
    config['dataset_path'],
    engine="zarr",
    chunks="auto",  # Load the data as a Dask array
)

def get_dates():
    times = dataset.get_index("time")
    date = times[0]
    end_date = times[-1]
    date_list = []

    while date <= end_date:
        date += timedelta(days=1)
        date_list.append(date)
        
    return date_list

def get_random_state():
    return choice([False] * 8 + [True] * 2)

def get_crop(input_slice, target_slice):
    # roughly over the mainland UK
    rand_x = randrange(300, 1400 - 128)
    rand_y = randrange(300, 850 - 128)

    # make a data selection
    selection = input_slice.isel(
        x=slice(rand_x, rand_x + 128),
        y=slice(rand_y, rand_y + 128),
    )

    # get the OSGB coordinate data
    osgb_data = np.stack(
        [
            selection["x_osgb"].values,
            selection["y_osgb"].values
        ]
    )

    if osgb_data.shape != (2, 128, 128):
        return None

    # get the input satellite imagery
    input_data = selection["data"].values
    if input_data.shape != (12, 128, 128):
        return None

    # get the target output
    target_output = (
        target_slice["data"]
        .isel(
            x=slice(rand_x, rand_x + 128),
            y=slice(rand_y, rand_y + 128),
        )
        .values
    )

    if target_output.shape != (24, 128, 128):
        return None

    return osgb_data, input_data, target_output

def process_time(start_time, filename):
    data_slice = dataset.loc[
        {
            "time": slice(
                start_time,
                start_time + timedelta(hours=2, minutes=55),
            )
        }
    ]
    if data_slice.sizes["time"] != 36:
        # print(start_time, data_slice.sizes["time"])
        return
        
    input_slice = data_slice.isel(time=slice(0, 12))
    target_slice = data_slice.isel(time=slice(12, 36))
    
    for crop_num in range(5):
        crop = get_crop(input_slice, target_slice)
        if crop == None:
            print(filename, 'none')
            continue

        osgb_data, input_data, target_output = crop
        path = os.path.join(config['data_path'], filename)

        np.save(path + '_' + str(crop_num) + '_osgb', osgb_data)
        np.save(path + '_' + str(crop_num) + '_input', input_data)
        np.save(path + '_' + str(crop_num) + '_target', target_output)

def process_day(date):
    print('processing ', date)
    start_time = time(9, 0)
    end_time = time(14, 0)
    is_valid = get_random_state()
    
    current_time = datetime.combine(date, start_time)
    while current_time.time() <= end_time:
        if is_valid:
            filename = 'valid\\' + '_'.join(str(current_time).split(' ')).replace(':', '-')
        else:
            filename = 'train\\' + '_'.join(str(current_time).split(' ')).replace(':', '-')

        process_time(current_time, filename)
        current_time += timedelta(minutes=60)

    date += timedelta(days=1)


if __name__ == '__main__':
    pool = Pool(processes=8)
    date_list = get_dates()
    pool.map(process_day, date_list)
    pool.close()
    pool.join()
