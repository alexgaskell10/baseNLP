import os
import sys
import pandas as pd
import numpy as np


def process_autoCW():
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

def process_autoQA():
    os.chdir('../go')
    df_orig = pd.read_csv('all_ads_cleaned_5_section_split.csv')
    OUTPATH = 'autoQA'

    # Create df containing original and cleaned ads
    # df_orig['cleaned_optimised'] = df_orig.section_1 + '\n' + df_orig.section_2 + '\n' + df_orig.section_3 + '\n' \
    #     + df_orig.section_4 + '\n' + df_orig.section_5
    df_orig['cleaned_optimised'] = df_orig.section_1 + df_orig.section_2 + df_orig.section_3 + \
        df_orig.section_4 + df_orig.section_5
    df = df_orig[['cleaned_optimised', 'original_cleaned']]

    # Split into train test val
    n = len(df)
    splits = {'test': 0.2, 'train': 0.7, 'dev': 0.1}
    shuf_idx = np.random.choice(n, n, replace=False)
    dfs = {
        'test': df.iloc[shuf_idx[ : int(splits['test']*n) ]], 
        'train': df.iloc[shuf_idx[ int(splits['test']*n) : int(splits['train']*n) ]], 
        'dev': df.iloc[shuf_idx[ -int(splits['dev']*n) : ]],
    } 

    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)

    for k, df_ in dfs.items():
        # Format correctly
        df_ = df_.melt()
        df_.columns = ['label', 'ad']
        df_['label'] = df_.label.str.replace('cleaned_optimised', '1').replace('original_cleaned', '0')
        df_['label'] = df_['label'].astype(int)
        df_ = df_.iloc[np.random.choice(len(df_), len(df_), replace=False)]
        df_.to_csv(os.path.join(OUTPATH, k + '.tsv'), sep='\t', index=True, header=False)

def process_adzuna_aspen():
    os.chdir('../go')
    df_aspen = pd.read_csv('third_party/aspen/aspen_sample_ads.csv')
    df_adzuna = pd.read_csv('third_party/aspen/aspen_sample_ads.csv')
    
    df = pd.concat([df_aspen[['description']], df_adzuna[['description']]]).reset_index(drop=True)
    df.columns = ['Description']
    df['label'] = 0
    df = df.reindex(columns=['label', 'Description'])
    df.to_csv('third_party/ads.tsv', sep='\t', index=True, header=False)
    if not os.path.exists('third_party/small'):
        os.mkdir('third_party/small')
    df.iloc[:10].to_csv('third_party/small/ads.tsv', sep='\t', index=True, header=False)

if __name__ == '__main__':
    # process_autoCW()
    # process_autoQA()    
    process_adzuna_aspen()
