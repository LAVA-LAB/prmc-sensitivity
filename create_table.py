import argparse
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path


def simplify_instance(instance):
    
    print('Try to simplify instance:', instance)
    
    dic = {
        'brp16_2': 'BRP (16,2)',
        'brp32_3': 'BRP (32,3)',
        'brp64_4': 'BRP (64,4)',
        'brp512_5': 'BRP (512,5)',
        'brp1024_6': 'BRP (1024,6)',
        'crowds3_5': 'Crowds (3,5)',
        'crowds6_5': 'Crowds (6,5)',
        'crowds10_5': 'Crowds (10,5)',
        'nand2_4': 'NAND (2,4)',
        'nand5_10': 'NAND (5,10)',
        'nand10_15': 'NAND (10,15)',
        'virus': 'Virus',
        'wlan0_param': 'WLAN0',
        'csma2_4_param': 'CSMA (2,4)',
        'coin4': 'Coin (4)',
        'maze_simple_extended_m5': 'Maze',
        'pomdp_drone_4-2-mem1-simple': 'Drone (mem1)',
        'pomdp_drone_4-2-mem5-simple': 'Drone (mem5)',
        'pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06': 'Satellite (36,5)',
        'pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40': 'Satellite(36,65',
        }
    
    for d,v in dic.items():
        if d in instance:
            simplified = v
            return simplified
        
    print('>>> Warning: Could not simplify instance name "{}"'.format(instance))
    return instance


def load_csv_files(mypath, simplify = True):

    # Get all files in provided folder
    filenames = next(os.walk(mypath), (None, None, []))[2]  # [] if no file
    
    print('- Nr of files found: {}'.format(len(filenames)))
    if len(filenames) == 0:
        print('>> No files found.')
        sys.exit()
    
    df = {}
    
    for i,file in enumerate(filenames):
        if Path(file).suffix != '.json':
            continue
        
        print('-- Read file "{}"'.format(file))
        df[i] = pd.read_json(os.path.join(mypath, file), typ='series')
        
        # Determine the date at which this file was created
        date_created_list = Path(file).stem.split('_')[-6:]
        df[i]['Date'] = datetime(*map(int, date_created_list))
        print('--- Created on:',df[i]['Date'])
        
        if 'Instance' in df[i]:
            instance = Path(df[i]['Instance']).stem
            if simplify:
                df[i]['Instance'] = simplify_instance(instance)
            else:
                df[i]['Instance'] = instance
        
    df_merged = pd.concat(df, axis=1).T
        
    return df_merged


if __name__ == "__main__":
    
    pd.set_option('display.max_columns', 100)
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    parser = argparse.ArgumentParser(description="Export results to tables as in paper")
    
    # Path to PRISM model to load
    parser.add_argument('--folder', type=str, action="store", dest='folder', 
                        default='output/slipgrid/', help="Folder to combine output files from")
    parser.add_argument('--table_name', type=str, action="store", dest='table_name', 
                        default='output/export_{}'.format(dt), help="Name of table csv file")
    parser.add_argument('--mode', type=str, action="store", dest='mode', 
                        default='detailed', help="Style of the table to export to")
    
    args = parser.parse_args()    
    
    assert args.mode in ['benchmark', 'gridworld', 'detailed']
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mypath = os.path.join(root_dir, args.folder)
    
    print('- Path to search:',mypath)
    if args.mode == 'gridworld':
        df_merged = load_csv_files(mypath, False)
    else:
        df_merged = load_csv_files(mypath, True)
    
    #####
    
    # Possible values for the number of derivatives
    K = np.sort(np.unique(df_merged['Num. derivatives']))    

    # Start by selecting only those rows for k=1 derivatives
    df_K0 = df_merged[ df_merged['Num. derivatives'] == K[0] ] 
                    
    df_out_add = pd.DataFrame({
        'Model verify [s]': df_K0['Model verify [s]'] + df_K0['Compute LHS matrices [s]'],
        'Compute one derivative [s]': df_K0['Solve one derivative [s]'] + df_K0['Compute LHS matrices [s]'] + df_K0['Compute RHS matrices [s]'] / df_K0['Parameters'],
        'Compute all derivatives [s]': df_K0['Solve all derivatives [s]'] + df_K0['Compute LHS matrices [s]'] + df_K0['Compute RHS matrices [s]']
        })
    
    # Model statistics and Problem 1 (differentiate all parameters explicitly)
    df_outa = pd.concat([ df_K0[['Date', 'Instance', 'Type', 'States', 'Parameters', 'Transitions', 'Solution']], df_out_add], axis=1)
    
    keys = ['Max. derivatives', 'Difference %'] #'Max. validation',
    
    keys_add = []
    for key in keys:
      if key in df_K0:
        keys_add += [key]
    
    df_outb = df_K0[keys_add]

    for k in K:
        
        # Define a new Pandas series
        series = pd.Series(index=df_K0.index, dtype=float)
        series.name = 'Problem 3, k={} [s]'.format(k)
        
        for index, row in df_K0.iterrows():    
            
            # Now add to this the corresponding entries for other values of k
            df_masked = df_merged[(df_merged['Instance'] == row['Instance']) &
                                  (df_merged['Type'] == row['Type']) &
                                  (df_merged['States'] == row['States']) &
                                  (df_merged['Parameters'] == row['Parameters']) &
                                  (df_merged['Num. derivatives'] == k)]
            
            if len(df_masked) == 0:
                continue
            
            row_masked = df_masked.iloc[0]
            
            # Compute time to solve problem 3 for this value of k
            series[index] = row_masked['Derivative LP (build and solve) [s]'] + row_masked['Compute LHS matrices [s]'] + row_masked['Compute RHS matrices [s]']
            
        df_outa = pd.concat([df_outa, series], axis=1)

    #####
        
    if args.mode != 'benchmark':
        # If not benchmark, add the columns related to the perturbation analysis
        # (for verifying the obtained derivatives)
        df_out = pd.concat([df_outa, df_outb], axis=1)
        
    else:
        df_out = df_outa

    if args.mode == 'gridworld':
        
        # States are off by one
        df_out['States'] += 1
        
        # For the gridworld experiments, drop the instance column (it's the same anyways)
        df_out = df_out.drop('Instance', axis=1)
        
        # Sort by 1) states and 2) parameters
        df_out.sort_values(['Type', 'States', 'Parameters', 'Transitions'],
                            ascending = [True, True, True, True],
                            inplace = True)
        
    else:
        
        # Sort by 1) states and 2) parameters
        df_out.sort_values(['Type', 'Date', 'States', 'Parameters', 'Transitions'],
                            ascending = [True, True, True, True, True],
                            inplace = True)
    
    # Drop create-on-date column (only used for sorting the DataFrame)
    df_out = df_out.drop('Date', axis=1)
    
    #####
    
    df_out.to_csv(os.path.join(root_dir, args.table_name + '.csv'))

    # Round entries for csv table
    for k in K:
        df_out['Problem 3, k={} [s]'.format(k)] = df_out['Problem 3, k={} [s]'.format(k)].map('{:,.2f}'.format)
    
    df_out['Solution'] = df_out['Solution'].map('{:,.2f}'.format)
    df_out['Model verify [s]'] = df_out['Model verify [s]'].map('{:,.2f}'.format)
    df_out['Compute all derivatives [s]'] = df_out['Compute all derivatives [s]'].map('{:,.2f}'.format)
    df_out['Compute one derivative [s]'] = df_out['Compute one derivative [s]'].map('{:,.2f}'.format)
    
    if 'Max. derivatives' in df_out:
      if type(df_out['Max. derivatives'][df_out.index[0]]) == float:
        if 'Difference %' in df_out:
            df_out['Difference %'] = df_out['Difference %'].map('{:,.1f}'.format)
        if 'Max. derivatives' in df_out:
            df_out['Max. derivatives'] = df_out['Max. derivatives'].map('{:,.2e}'.format)
        if 'Max. validation' in df_out:
            df_out['Max. validation'] = df_out['Max. validation'].map('{:,.2e}'.format)
    
    print('- All files merged into a single Pandas DataFrame')

    # In LaTeX table, leave out the column for computing one derivative (we only show all derivs.)
    df_out_tex = df_out.drop('Compute one derivative [s]', axis=1)
    
    df_out_tex.to_latex(os.path.join(root_dir, args.table_name + '.tex'),
                       index=False)
    
    print('- Exported to CSV and LaTeX table')