import csv
from collections import defaultdict
import sys

def main(dir_name, num_of_entity):
    columns = defaultdict(list) 

    with open('../data/'+dir_name+'/'+num_of_entity+'_finetuned_'+str(checkpoint_dir)+'.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: 
            for (k,v) in row.items(): 
                columns[k].append(v) 
                                    


    total_score = 0
    for score in columns['avg_binary_score']:
        total_score += float(score)

    average_score = total_score / len(columns['avg_binary_score'])
    # print(total_score)
    # print(len(columns['avg_binary_score']))
    print(average_score)

if __name__ == "__main__":
    dir_name = sys.argv[1]
    num_of_entity = sys.argv[2]
    checkpoint_dir = sys.argv[3]
    main(dir_name, num_of_entity)