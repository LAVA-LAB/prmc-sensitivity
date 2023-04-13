import argparse
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

def load_csv_files(mypath):

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
        
    df_merged = pd.concat(df, axis=1).T
        
    return df_merged

if __name__ == "__main__":
    
    pd.set_option('display.max_columns', 100)
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    parser = argparse.ArgumentParser(description="Program to compute gradients for prMDPs")
    
    # Path to PRISM model to load
    parser.add_argument('--folder', type=str, action="store", dest='folder', 
                        default='output/slipgrid/', help="Folder to combine output files from")
    parser.add_argument('--table_name', type=str, action="store", dest='table_name', 
                        default='output/export_{}'.format(dt), help="Name of table csv file")
    parser.add_argument('--mode', type=str, action="store", dest='mode', 
                        default='detailed', help="Style of the table to export to")
    
    args = parser.parse_args()    
    
    args.folder = 'output/benchmarks_cav23_partial'
    args.table_name = 'output/benchmarks_cav23_partial'
    args.mode = 'gridworld'
    
    assert args.mode in ['benchmark', 'gridworld', 'detailed']
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mypath = os.path.join(root_dir, args.folder)
    
    print('- Path to search:',mypath)
    df_merged = load_csv_files(mypath)
    
    #####
    
    # Possible values for the number of derivatives
    K = np.sort(np.unique(df_merged['Num. derivatives']))    

    # Start by selecting only those rows for k=1 derivatives
    df_out = df_merged[ df_merged['Num. derivatives'] == K[0] ] 
    
    # Model statistics and Problem 1 (differentiate all parameters explicitly)
    df_outa = df_out[['Instance', 'Type', 'States', 'Parameters', 'Transitions', 'Solution', 'Model verify [s]',
                     'Differentiate explicitly [s]']]
    
    df_outb = df_out[['Max. derivatives', 'Max. validation', 'Difference %']]

    for k in K:
        
        # Define a new Pandas series
        series = pd.Series(index=df_out.index, dtype=float)
        series.name = 'Problem 3, k={} [s]'.format(k)
        
        for index, row in df_out.iterrows():    
            
            # Now add to this the corresponding entries for other values of k
            df_masked = df_merged[(df_merged['Instance'] == row['Instance']) &
                                  (df_merged['Type'] == row['Type']) &
                                  (df_merged['Num. derivatives'] == k)]
            
            if len(df_masked) == 0:
                continue
            
            row_masked = df_masked.iloc[0]
            
            # Compute time to solve problem 3 for this value of k
            problem3_time = row_masked['LP (define matrices) [s]'] + row_masked['LP (solve) [s]']
            series[index] = np.round(problem3_time, 2)
            
        df_outa = pd.concat([df_outa, series], axis=1)
        
    #####
        
    if args.mode != 'benchmark':
        # If not benchmark, add the columns related to the perturbation analysis
        # (for verifying the obtained derivatives)
        df_out = pd.concat([df_outa, df_outb], axis=1)
        
    if args.mode == 'gridworld':
        # States are off by one
        df_out['States'] += 1
        
        # For the gridworld experiments, drop the instance column (it's the same anyways)
        df_out = df_out.drop('Instance', axis=1)
        
    #####
        
    # Sort by 1) states and 2) parameters
    df_out.sort_values(['Type', 'States', 'Parameters', 'Transitions'],
                        ascending = [True, True, True, True],
                        inplace = True)
    
    #####
    
    # Round to 1 decimal places
    df_out['Solution'] = df_out['Solution'].map('{:,.2f}'.format)
    df_out['Model verify [s]'] = df_out['Model verify [s]'].map('{:,.2f}'.format)
    df_out['Differentiate explicitly [s]'] = df_out['Differentiate explicitly [s]'].map('{:,.2f}'.format)
    
    if type(df_out['Max. derivatives'][df_out.index[0]]) == float:
        if 'Difference %' in df_out:
            df_out['Difference %'] = df_out['Difference %'].map('{:,.1f}'.format)
        if 'Max. derivatives' in df_out:
            df_out['Max. derivatives'] = df_out['Max. derivatives'].map('{:,.2e}'.format)
        if 'Max. validation' in df_out:
            df_out['Max. validation'] = df_out['Max. validation'].map('{:,.2e}'.format)
    
    print('- All files merged into a single Pandas DataFrame')
    
    df_out.to_csv(os.path.join(root_dir, args.table_name + '.csv'))
    df_out.to_latex(os.path.join(root_dir, args.table_name + '.tex'),
                       index=False)
    
    print('- Exported to CSV and LaTeX table')