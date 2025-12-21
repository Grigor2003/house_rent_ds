import pandas as pd
import numpy as np

df = pd.read_csv('Data/rent.csv')

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    return df[(df[column] >= Q1) & (df[column] <= Q3)]

df_clean = remove_outliers(df, 'Size')
df_clean = remove_outliers(df_clean, 'Rent')

print(f"Original: {len(df)} rows")
print(f"After outlier removal: {len(df_clean)} rows")
print(f"Size range: {df_clean['Size'].min()} - {df_clean['Size'].max()}")
print(f"Rent range: {df_clean['Rent'].min()} - {df_clean['Rent'].max()}")

df_clean[['Size', 'Rent']].to_csv('size_rent_clean.dat', sep=' ', index=False, header=False)
print("Data saved to size_rent_clean.dat")



