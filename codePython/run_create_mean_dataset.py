import pandas as pd

FILENAME="kdd99_10percent_10_fold_2021-01-23"


df_all = pd.read_csv(f"./results/{FILENAME}.csv")

mean_columns = [ column_name for column_name in df_all.columns if "mean_" in column_name]
start_columns = [ column_name for column_name in df_all.columns if "start" in column_name]
end_columns = [ column_name for column_name in df_all.columns if "end" in column_name]
#print(mean_columns)
columns_to_be_dropped = ['cross_validation_count'] + mean_columns + start_columns + end_columns

df_all.drop(columns=columns_to_be_dropped,inplace=True)

df = df_all.groupby(['classifier_name']).mean().add_prefix('mean_').reset_index()

print(df.columns)

df.to_csv(f"./results/mean_{FILENAME}.csv",index=False,float_format='%8.4f')