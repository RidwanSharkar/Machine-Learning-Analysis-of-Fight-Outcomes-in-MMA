import pandas as pd

file_path = 'data/masterMLpublic.csv'
try:
    df = pd.read_csv(file_path, nrows=45) 
    selected_data = df.iloc[:, 3:13]
    print(selected_data)
except FileNotFoundError:
    print("File not found. Check the file path and ensure it is correct.")
except Exception as e:
    print("An error occurred:", e)
