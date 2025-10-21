# %%
# %load_ext autoreload
# %autoreload 2
import dbcgen
import pandas as pd
filein='input/test_example2.dbc.xlsx'

# %%
# read signals sheet
df=pd.read_excel(filein, sheet_name='signals')
df.head()

# %%
# prepare data for dbcgen
df_dbc = dbcgen.generate_dbc_data(df)
# Add to Excel file
df_dbc.head()

# %%
with pd.ExcelWriter(filein, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:
    df_dbc.to_excel(writer, sheet_name="DBC_Export", index=False)

print(f"Added DBC export to {filein} in sheet DBC_Export")

# %%
# generate dbc file content
dbcgen.generate_dbc_from_df(df_dbc, output_dbc_file="output/output.dbc")


