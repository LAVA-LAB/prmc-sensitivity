# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/parse_output"

import argparse
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

parser = argparse.ArgumentParser(description="Program to compute gradients for prMDPs")

# Path to PRISM model to load
parser.add_argument('--folder', type=str, action="store", dest='folder', 
                    default='output/', help="Folder to combine output files from")
parser.add_argument('--table_name', type=str, action="store", dest='table_name', 
                    default='tables/export_{}'.format(dt), help="Name of table csv file")

args = parser.parse_args()    

root_dir = os.path.dirname(os.path.abspath(__file__))
mypath = os.path.join(root_dir, args.folder)

print('- Path to search:',mypath)

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

print('- All files merged into a single Pandas DataFrame')

df_merged.to_csv(os.path.join(root_dir, args.table_name + '.csv'))
df_merged.to_latex(os.path.join(root_dir, args.table_name + '.tex'))

print('- Exported to CSV and LaTeX table')