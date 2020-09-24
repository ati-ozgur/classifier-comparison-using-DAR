import pandas as pd


def normalize_min_max(df,column_name):
	new_column_name = column_name.replace("mean_","weighted_")
	min_val = df[column_name].min()
	max_val = df[column_name].max()
	df[new_column_name] = (df[column_name] - min_val) / (max_val - min_val)
	if not "accuracy" in column_name:
		df[new_column_name] = 1 - df[new_column_name]

df = pd.read_csv("./results/kdd99_10percent_10_fold_2020-09-15.csv")

weight_accuracy_score = 40
weight_model_size = 30
weight_training_time = 10
weight_testing_time = 20

total_weight = weight_accuracy_score + weight_model_size + weight_training_time + weight_testing_time

assert total_weight == 100


print(df.columns)

for column_name in df.columns:
	if "mean_" in column_name:
		normalize_min_max(df,column_name)


df["mean_accuracy_score"] = df["mean_accuracy_score"] * 100
df["mean_training_time"] = df["mean_training_time"] 
df["mean_testing_time"] = df["mean_testing_time"] 
df["mean_model_size"] = df["mean_model_size"] / 1024

df["Total"] = weight_accuracy_score * df["weighted_accuracy_score"] + weight_testing_time* df["weighted_testing_time"] + weight_training_time* df["weighted_training_time"] + weight_model_size* df["weighted_model_size"]

#print(df[["classifier_name","Total"]])

df.sort_values(by=["Total"],ascending =False, inplace=True)

output = ""

output+="\\textbf{Classifier Name} & \\textbf{Accuracy (\\%)}  & \\textbf{Model Size(Mb)}  & \\textbf{Training Time (sec)} & \\textbf{Testing Time (sec)} & \\textbf{Total}\\\\"
output += "\n" 
output += "\\midrule \n"
output +=f"Weights & {weight_accuracy_score} & {weight_model_size} & {weight_training_time} & {weight_testing_time} & 100  \\\\"
output += "\n"
output += "\\midrule \n"

for index, row in df.iterrows():
	classifier_name = row["classifier_name"]
	mean_accuracy_score = row["mean_accuracy_score"]
	mean_training_time = row["mean_training_time"]
	mean_testing_time = row["mean_testing_time"]
	mean_model_size = row["mean_model_size"]
	total = row["Total"]

	output+=f" {classifier_name} &"
	output+=f" {mean_accuracy_score:.2f} &"
	output+=f" {mean_model_size:.2f}  & "
	output+=f" {mean_training_time:.2f} & " 
	output+=f" {mean_testing_time:.2f} & "
	output+=f" {total:.2f} \\\\ \n"


output += "\\bottomrule \n"


tabular_filename = "../latex/table-DAR-KDD99-results-tabular.tex"

f = open(tabular_filename, "w")
f.write(output)
f.close()


print(f"tabular_filename: {tabular_filename} saved")

