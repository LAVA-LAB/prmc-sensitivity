import argparse
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from ast import literal_eval

if __name__ == "__main__":
    
    '''
    Gather data from tables specified in the argument --files, and export the
    data used in the scatter plots of the paper
    '''
    
    print('\nStart exporting data for scatter plots...')
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create data for scatter plots")
    parser.add_argument('--files', type=str, action="store", dest='files', 
                        default=None, help="CSV files to merge for scatter plots")
    args = parser.parse_args()    
    
    if args.files is not None:
        args.files = literal_eval(args.files)
        assert type(args.goal_label) == list
        
    args.files = ['output/slipgrid_partial_table.csv', 'output/benchmarks_partial_table.csv']
    
    # Load the provided CSV files
    df = {}
    for i,file in enumerate(args.files):
        df[i] = pd.read_csv(os.path.join(root_dir, file))
    
    print('- Loaded data from {} .csv files: {}'.format(len(df), args.files))    
    
    df_merged = pd.concat(df, axis=0)

    # Export csv files for the scatter plots in the paper
    time_verify_vs_differentiating = df_merged[['Type', 'Model verify [s]', 'Compute one derivative [s]', 'Problem 3, k=1 [s]', 'Parameters']]
    time_verify_vs_differentiating.columns = ['Type', 'Verify', 'OneDeriv', 'Highest', 'Parameters']
    time_verify_vs_differentiating = time_verify_vs_differentiating.round(4)

    time_verify_vs_differentiating.to_csv(os.path.join(root_dir, 'output/scatter_time_verify_vs_differentiating.csv'), index=False, sep=';')

    time_highest1_vs_all = df_merged[['Type', 'Problem 3, k=1 [s]', 'Compute all derivatives [s]', 'Parameters']]
    time_highest1_vs_all.columns = ['Type', 'Highest', 'All', 'Parameters']
    time_highest1_vs_all = time_highest1_vs_all.round(4)
    time_highest1_vs_all = time_highest1_vs_all.dropna(axis=0)
    
    time_highest1_vs_all.to_csv(os.path.join(root_dir, 'output/scatter_time_highest1_vs_all.csv'), index=False, sep=';')
    
    time_highest10_vs_all = df_merged[['Type', 'Problem 3, k=10 [s]', 'Compute all derivatives [s]', 'Parameters']]
    time_highest10_vs_all.columns = ['Type', 'Highest', 'All', 'Parameters']
    time_highest10_vs_all = time_highest10_vs_all.round(4)
    time_highest10_vs_all = time_highest10_vs_all.dropna(axis=0)
    
    time_highest10_vs_all.to_csv(os.path.join(root_dir, 'output/scatter_time_highest10_vs_all.csv'), index=False, sep=';')
    
    print('- Exported data for 3 scatter plots')