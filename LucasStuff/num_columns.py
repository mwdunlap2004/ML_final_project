import pandas as pd
import numpy as np

df1 = pd.read_csv('../combined_data.csv')
df2 = pd.read_csv('../categorized_data.csv')
df3 = pd.read_csv('../final_data.csv')

print(f'Categorized has {len(df2.columns)} cols\nCombined has {len(df1.columns)} cols\nFinal \
final has {len(df3.columns)} cols')
