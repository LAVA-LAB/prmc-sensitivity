# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/create_table.py"

import argparse
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

pd.set_option('display.max_columns', 100)

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

parser = argparse.ArgumentParser(description="Program to compute gradients for prMDPs")

# Path to PRISM model to load
parser.add_argument('--folder', type=str, action="store", dest='folder', 
                    default='output/slipgrid/', help="Folder to combine output files from")
parser.add_argument('--table_name', type=str, action="store", dest='table_name', 
                    default='tables/export_{}'.format(dt), help="Name of table csv file")
parser.add_argument('--detailed', dest='detailed', action='store_true',
                    help="If True, output more columns in table.")
parser.set_defaults(detailed=False)

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

# Round to 3 decimal places
round_keys = ['Solution', 'Model verify [s]', 'LP (solve) [s]', 'Difference %', 
              'Differentiate explicitly [s]']

for key in round_keys:
    if key in df_merged:
        df_merged[key] = df_merged[key].map('{:,.3f}'.format)
    else:
        print('Warning, key `{}` not in dataframe'.format(key))

df_merged['Max. derivatives'] = df_merged['Max. derivatives'].map('{:,.2e}'.format)
if 'Max. validation' in df_merged:
    df_merged['Max. validation'] = df_merged['Max. validation'].map('{:,.2e}'.format)

print(df_merged)

# Sort by 1) states and 2) parameters
df_merged.sort_values(['Type', 'States', 'Parameters', 'Transitions'],
                      ascending = [True, True, True, True],
                      inplace = True)

# Only keep certain columns for Latex table
if args.detailed:
    table_columns = ['instance']
else:
    table_columns = []
table_columns +=['Type', 
                 'States', 
                 'Parameters',
                 'Transitions',
                 'Solution',
                 'Model verify [s]']

if 'Differentiate explicitly [s]' in df_merged:
    table_columns += ['Differentiate explicitly [s]']
    
table_columns += ['LP (solve) [s]',
                 'Max. derivatives', 
                 'Max. validation', 
                 'Difference %']

print('- All files merged into a single Pandas DataFrame')

df_merged.to_csv(os.path.join(root_dir, args.table_name + '.csv'))
df_merged.to_latex(os.path.join(root_dir, args.table_name + '.tex'),
                   columns=table_columns,
                   index=False)

print('- Exported to CSV and LaTeX table')