import os
import sys
import pandas as pd
import numpy as np

os.chdir('../go')
df_orig = pd.read_csv('all_ads_cleaned_5_section_split.csv')
# ['Unnamed: 0', 'date', 'doc_to_html', 'optimised', 'original_cleaned',
#        'section_1', 'section_2', 'section_3', 'section_4', 'section_5']
sections = ['section_1', 'section_2', 'section_3', 'section_4', 'section_5']

# Drop rows which do not have all 5 sections
df = df_orig[~df_orig[sections].isna().any(axis=1)]

# Split into train test vals
n = len(df)
splits = {'test': 0.2, 'train': 0.7, 'val': 0.1}
shuf_idx = np.random.choice(n, n, replace=False)
dfs = {
    'test': df.iloc[shuf_idx[ : int(splits['test']*n) ]], 
    'train': df.iloc[shuf_idx[ int(splits['test']*n) : int(splits['train']*n) ]], 
    'val': df.iloc[shuf_idx[ -int(splits['val']*n) : ]],
} 

# Write as output files
for section in sections:
    if not os.path.exists(section):
        os.mkdir(section)
    
    for k, df_ in dfs.items():
        # Write source files
        with open(os.path.join(section, k + '.source'), 'w') as f:
            for line in df_['original_cleaned']:
                f.write(line.lstrip('.').lstrip() + '\n')
        # Write target files
        with open(os.path.join(section, k + '.target'), 'w') as f:
            for idx, line in enumerate(df_[section]):
                f.write(line.lstrip('.').lstrip() + '\n')

