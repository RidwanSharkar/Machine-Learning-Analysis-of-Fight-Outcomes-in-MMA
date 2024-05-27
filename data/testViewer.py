import pandas as pd

file_path = 'data/masterMLpublic.csv'
try:
    df = pd.read_csv(file_path, nrows=11) 
    #print(f"The number of columns in the CSV file is: {len(df.columns)}")

    selected_data = df.iloc[:, 54:62]
    print(selected_data)
except FileNotFoundError:
    print("File not found. Check the file path and ensure it is correct.")
except Exception as e:
    print("An error occurred:", e)
