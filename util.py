import pandas as pd
import csv


def add_id():
    df = pd.read_csv('data/diabetes/diabetes.csv', sep=',')
    # Add a new column named 'ID' with the line number for each row
    # df = df.drop('ID', axis=1)
    df.insert(0, 'ID', range(1, len(df) + 1))
    # Add an increasing index to the last column
    # df = df.drop('ID', axis=1)
    # df.iloc[:, -1] = df.iloc[:, -1].astype(str) + '_' + (df.index + 1).astype(str)
    df.to_csv('diabetes_id_first.csv', index=False)


def merge_df():
    # k_val = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for k in [2, 5, 10, 20, 50, 70, 100]:
        # for l in [2, 3]:
        # Read the first CSV file
        # df1 = pd.read_csv('data/adult/adult_inc_ID.csv', sep=',')
        df1 = pd.read_csv("data/diabetes/diabetes_id_first.csv", sep=',')

        # Read the second CSV file
        df2 = pd.read_csv(f"anonymized_data/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv", sep=',')

        # Perform a right join between the two DataFrames on a common column
        merged_df = pd.merge(df1, df2, on='ID', how='right')

        # Save the merged DataFrame to CSV
        merged_df.to_csv(f"results/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv", index=False)


def fix_cols():
    for k in [2, 5, 10, 20, 50, 70, 100]:
        # for l in [2, 3]:
        df = pd.read_csv(f"anonymized_data/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv",
                         usecols=lambda x: "_x" not in x)
        df.to_csv(f"results/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv", index=False)


def fix_header():
    # Define the string to replace the first line with
    # new_first_line = "workclass,fnlwgt,educational-num,relationship,gender,capital-gain,capital-loss,hours-per-week,native-country,ID,age,education,marital-status,occupation,race,income".split(
    #     ',')

    # new_first_line = "ID,mean_rooms,mean_bedrooms,population,households,longitude,latitude,housing_median_age,median_income,median_house_value,ocean_proximity".split(
    #     ',')
    new_first_line = "ID,gender,hypertension,smoking_history,bmi,age,heart_disease,HbA1c_level,blood_glucose_level,diabetes".split(
        ',')
    # Open the CSV file for reading and writing
    for k in [2, 5, 10, 20, 50, 70, 100]:
        for l in [2, 3]:
            # with open(f"anonymized_data/adult/mondrian/anon_ID_full/anonymized_{k}.csv", 'r+', newline='\n') as csvfile:
            with open(f"anonymized_data/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv", 'r+',
                      newline='\n') as csvfile:
                reader = csv.reader(csvfile)
                # Replace the first line with the new string
                lines = list(reader)
                lines[0] = new_first_line
                csvfile.seek(0)
                writer = csv.writer(csvfile)
                writer.writerows(lines)


def split_id():
    for k in [2, 5, 10, 20, 50, 70, 100]:
        # for l in [2, 3]:
        df = pd.read_csv(f"anonymized_data/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv", sep=';')
        # Extract the last column
        last_column = df.iloc[:, -1]

        # Apply split function and extract second value
        sens = []
        ids = []
        for value in last_column:
            tmp = value.split('_')
            sens.append(tmp[0])
            ids.append(tmp[1])

        # Add 'ID' as the first column to the DataFrame
        df.insert(0, 'ID', ids)
        df = df.drop('diabetes', axis=1)
        df['diabetes'] = sens
        df.to_csv(f"results/diabetes/mondrian/anonypy/diabetes_anon_k_{k}.csv")


# df = pd.read_csv(f"data/diabetes/diabetes_id_first.csv", sep=',')
# # Count the occurrences of each class
# class_counts = df['diabetes'].value_counts()
#
# # Calculate the total number of samples
# total_samples = class_counts.sum()
#
# # Calculate the percentage of each class
# positive_percentage = (class_counts[1] / total_samples) * 100
# negative_percentage = (class_counts[0] / total_samples) * 100
#
# # Print the percentage of each class
# print("Positive diabetes class percentage: {:.2f}%".format(positive_percentage))
# print("Negative diabetes class percentage: {:.2f}%".format(negative_percentage))

with open("performances/poker_hand/performance_weighted", "r") as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    if i % 2 == 0:
        if line.startswith("f1:"):
            accuracy = line.split(":")[1].strip().split()[0]
            print(round(float(accuracy), 2))

# 0
# add_id()
# 1
# split_id()
# 2
# merge_df()
# 3
# fix_cols()
# 4
# fix_header()
