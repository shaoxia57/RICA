import sys
import json

axioms = []
file_name = sys.argv[1]
with open(file_name, 'r') as f:
    data = json.load(f)
    for i in data:
        axioms.append(data[i]['original']['original'])
        
with open('axioms_collected.txt', 'a') as f:
    for a in axioms:
        f.write(a+"\n")

