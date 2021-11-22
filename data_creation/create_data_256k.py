import json
from tqdm import tqdm


with open('../data/256k_result_corrected.tsv', 'r') as f:
    lines = f.readlines()
    rica_256k_statements = []
    for line in lines:
        rica_256k_statements.append(line.split('\t')[1].strip('\n'))

rica_10k = []
with open('../data/truism_data/RICA_10k_probe_sets.json','r') as f:
    data = json.load(f)
    perturbations = ['original', 'negation']
    asymetric = ['original', 'asymmetric_premise', 'asymmetric_conclusion']
    for i in data:
        for pert in perturbations:
            for order in asymetric:
                if len(data[i][pert][order]) != 0:
                    statement = data[i][pert][order][0]
                    reverse = data[i][pert][order][1]
                    rica_10k.append(statement)
                    rica_10k.append(reverse)


pairs_256k = []
more_count=0
less_count=0
hotter_count=0
colder_count=0
not_likely_count=0
likely_count=0
better_count=0
worse_count=0
younger_count=0

in_rica_10k = 0
for statement in rica_256k_statements:
    reverse=''
    if ', so ' not in statement:
        print(statement+' DOES NOT HAVE , SO !!')
        continue
    if ' more ' in statement.split(', so ')[1]:
        more_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' more ',' less ')
        right_answer = 'more'
        wrong_answer = 'less'
        
    elif ' less ' in statement.split(', so ')[1]:
        less_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' less ',' more ')
        right_answer = 'less'
        wrong_answer = 'more'
        
    elif ' hotter ' in statement.split(', so ')[1]:
        hotter_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' hotter ',' colder ')
        right_answer = 'hotter'
        wrong_answer = 'colder'
        
    elif ' colder ' in statement.split(', so ')[1]:
        colder_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' colder ',' hotter ')
        right_answer = 'colder'
        wrong_answer = 'hotter'
        
    elif ' not likely ' in statement.split(', so ')[1]:
        not_likely_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' not likely ',' likely ')
        right_answer = 'not likely'
        wrong_answer = 'likely'
    elif ' likely ' in statement.split(', so ')[1]:
        likely_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' likely ',' not likely ')
        right_answer = 'likely'
        wrong_answer = 'not likely'
    elif ' better ' in statement.split(', so ')[1]:
        better_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' better ',' worse ')
        right_answer = 'better'
        wrong_answer = 'worse'
        
    elif ' worse ' in statement.split(', so ')[1]:
        worse_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' worse ',' better ')
        right_answer = 'worse'
        wrong_answer = 'better'
        
    elif ' younger ' in statement.split(', so ')[1]:
        #younger_count+=1
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' younger ',' older ')
        right_answer = 'younger'
        wrong_answer = 'older'
        
    elif ' harder ' in statement.split(', so ')[1]:
        reverse = statement.split(', so ')[0]+', so '+ statement.split(', so ')[1].replace(' harder ',' easier ')
        #younger_count+=1
        right_answer = 'harder'
        wrong_answer = 'easier'
        
    if reverse == '':
        print(statement)
        continue

    pairs_256k.append([statement,reverse, right_answer, wrong_answer])
    
        
rica_100k_dict = {}

for i in tqdm(range(len(pairs_256k))):
    
    pair = []
    if len(pairs_256k[i]) != 0:
        statement = pairs_256k[i][0]
        reverse = pairs_256k[i][1]
        right = pairs_256k[i][2]
        wrong = pairs_256k[i][3]
        if statement not in rica_10k or reverse not in rica_10k:
            pair.append([statement, reverse, right, wrong])
        else:
            in_rica_10k += 1
        
    if i < 104400:
        rica_100k_dict[str(i)] = pair

with open('../data/RICA_100k_probe_sets.json', 'w') as probe:
    json.dump(rica_100k_dict, probe, indent=4)




print("remove {} statements from 10k".format(in_rica_10k))

