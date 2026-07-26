import pandas as pd

path = "tde_dyn_results/orbital_params.csv"
df = pd.read_csv(path, usecols=[8])

col_name = df.columns[0]
max_val = df[col_name].max()
min_val = df[col_name].min()
mean_val = df[col_name].mean()

print("name:", col_name)
print(
    f"max value = {max_val:.3f} msun\nmin value = {min_val:.3f} msun\nmean value = {mean_val:.3f}msun"
)
