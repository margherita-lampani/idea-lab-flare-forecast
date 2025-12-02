"""
Script to download GOES data from Fido
"""
import astropy.units as u
from astropy.table import QTable
from datetime import datetime,timedelta
import os
import pandas as pd
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.base_client import QueryResponseTable
import sys
import yaml 

root = sys.argv[1]

# Read config and extract start and end year
with open('snakemake_config.yaml') as f:
    config = yaml.safe_load(f)

start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')

for year in range(start_date.year, end_date.year + 1):
    print(f"Downloading GOES data for year {year} ...")
    
    if not os.path.exists(root + str(year)):
        os.mkdir(root + str(year))
    
    # Determine start and end dates for this year
    if year == start_date.year:
        # First year: use the actual start date
        tstart = start_date
    else:
        # Other years: start from January 1st
        tstart = datetime(year, 1, 1)
    
    if year == end_date.year:
        # Last year: use the actual end date
        tend = end_date
    else:
        # Other years: go until December 31st
        tend = datetime(year, 12, 31)
    
    # create query
    result = Fido.search(a.Time(datetime.strftime(tstart, '%Y-%m-%d %H:%M'),
                                datetime.strftime(tend, '%Y-%m-%d %H:%M')),
                        a.Instrument('XRS'))
    
    if len(result[0]) == 0:   # no data found
        continue
    
    # filter only latest GOES satellite data
    df_result = result[0].to_pandas()
    
    if 'SatelliteNumber' in df_result.columns:
        df_result.sort_values(['Start Time', 'SatelliteNumber'], inplace=True)
    else:
        df_result.sort_values(['Start Time'], inplace=True)
    
    df_result.drop_duplicates('Start Time', keep='last', inplace=True)
    
    # remake into query 
    result_table = QTable.from_pandas(df_result)
    result_new = QueryResponseTable(result_table)
    result_new.client = result[0].client
    
    # download data
    file_goes = Fido.fetch(result_new, path=root + str(year))
    print(f"Downloaded {len(result_new)} files for year {year}")

# Old version without using config file
'''
for year in (1996, 2023): #in range(1980,2023)
    print(f"Downloading GOES data for year {year} ...")

    if not os.path.exists(root+str(year)):
        os.mkdir(root+str(year))

    # create query
    tstart=datetime(year,1,1)
    tend=datetime(year,12,31)
    result = Fido.search(a.Time(datetime.strftime(tstart,'%Y-%m-%d %H:%M'),
                                datetime.strftime(tend,'%Y-%m-%d %H:%M')),
                                a.Instrument('XRS'))
    
    if len(result[0])==0:   # no data found
        continue
    
    # filter only latest GOES satellite data
    df_result = result[0].to_pandas()

    #MODIFIED
    #df_result.sort_values(['Start Time','SatelliteNumber'],inplace=True)
    if 'SatelliteNumber' in df_result.columns:
        df_result.sort_values(['Start Time','SatelliteNumber'], inplace=True)
    else:
        df_result.sort_values(['Start Time'], inplace=True)

    df_result.drop_duplicates('Start Time',keep='last',inplace=True)
    # remake into query 
    result_table = QTable.from_pandas(df_result)
    result_new = QueryResponseTable(result_table)
    result_new.client = result[0].client

    # download data
    file_goes = Fido.fetch(result_new,path=root+str(year))
    print(f"Downloaded {len(result_new)} files for year {year}")
'''
