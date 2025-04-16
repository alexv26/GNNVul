import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Location of the current file
FILE = os.path.join(BASE_DIR, "all_valid_data_new.json")

try:
    with open(FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load the existing array
except FileNotFoundError:
    raise FileNotFoundError

for i in range(len(data)):
    cvss = float(data[i]["cvss_score"])
    label = 1 if cvss != 0.0 else 0
    data[i]["target"] = label

with open(FILE, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)