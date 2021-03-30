import pandas as pd

df = pd.read_csv('sample_ads.csv')

df.original.to_csv('data/test_GO/train.source', header=False, index=False)
df.original.to_csv('data/test_GO/test.source', header=False, index=False)
df.original.to_csv('data/test_GO/val.source', header=False, index=False)
df.optimised.to_csv('data/test_GO/train.target', header=False, index=False)
df.optimised.to_csv('data/test_GO/test.target', header=False, index=False)
df.optimised.to_csv('data/test_GO/val.target', header=False, index=False)