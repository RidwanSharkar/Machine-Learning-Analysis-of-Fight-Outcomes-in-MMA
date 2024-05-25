import pandas as pd
import json

file_path = 'data/mma-differentials-and-elo-metadata.json'

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(json.dumps(data, indent=4))  # This will print the JSON data with indentation for readability
except FileNotFoundError:
    print("File not found. Check the file path and ensure it is correct.")
except Exception as e:
    print("An error occurred:", e)

#file_path = 'data/masterMLpublic.csv'
#try:
#    df = pd.read_csv(file_path, nrows=45) 
#    selected_data = df.iloc[:, 3:13]
#    print(selected_data)
#except FileNotFoundError:
#    print("File not found. Check the file path and ensure it is correct.")
#except Exception as e:
#    print("An error occurred:", e)
