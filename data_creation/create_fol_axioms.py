import json


def physical_fol():
    physical_fol = '∀x∀y (Q(x,y) → R(x,y))'
    with open("../data/truism_data/physical_data_sentences_2.json", "r") as f:
        data = json.load(f)
        physical_sents = []
        for i in data:
            physical_sents.append(data[i]['original']['original'])
            
    for sent in physical_sents:
        premise = sent.split(',')[0]
        conclusion = sent.split(',')[1]

        premise_word = premise.split(' ')
        for i, word in enumerate(premise_word):
            if word == 'than':
                Q = premise_word[i-1].capitalize()+'Than'
        
        conclusion_word = conclusion.split(' ')
        R = ''
        for i, word in enumerate(conclusion_word):
            
            if word == 'more' or word == 'better' or 'er' in word or 'less' in word or 'worse' in word:
                if conclusion_word[i+1] != 'than':
                    for k in range(i, len(conclusion_word)):
                        if conclusion_word[k] != 'than':
                            R += conclusion_word[k].capitalize()
                        else:
                            break
                else:
                    R = conclusion_word[i-1].capitalize()+word.capitalize()+'Than'
            
            # else:
            #     if word == 'than':
            #         R = conclusion_word[i-1].capitalize()+'Than'
        physical_fol_axiom = physical_fol.replace('Q', Q)
        physical_fol_axiom = physical_fol_axiom.replace('R', R)
        fol_axioms.append(physical_fol_axiom)

def material_fol():
    material_fol = 'material(A, mat_type1) ^ material (B, mat_type2) → Q(f(A), f(B))'
    with open("../data/truism_data/material_data_2.json", "r") as f:
        data = json.load(f)
        material_sents = []
        for i in data:
            mat_type1 = data[i]['material_1']
            mat_type2 = data[i]['material_2']
            Q = data[i]['premise_switch']['0'][0]
            f = data[i]['antonym_switch'][0]
            material_fol_axiom = material_fol.replace('mat_type1', mat_type1).replace('mat_type2', mat_type2).replace('Q', Q).replace('f', f)
            fol_axioms.append(material_fol_axiom)
    
def social_fol():
    social_fol_one = 'f(A) ^ ¬f(B) → Q(g(A), g(B))' 
    social_fol_two = 'Q(A, B) → R(f(A), f(B))'
    social_fol_three = 'Q(f(A), f(B)) → Q(g(A), g(B))'

    with open("../data/truism_data/social_data_2.json", "r") as f:
        data = json.load(f)
        physical_sents = []
        for i in data:
            situation = data[i]['situation']
            if 'while' in situation:
                Q = data[i]['premise_switch']['0'][0]
                g = data[i]['antonym_switch'][0]
                social_fol_axiom = social_fol_one.replace('Q', Q).replace('g',g)
                fol_axioms.append(social_fol_axiom)
            elif 'B\'s' in situation:
                Q = situation.split('B\'s ')[1]
                R = data[i]['premise_switch']['0'][0]
                f = data[i]['antonym_switch'][0]
                social_fol_axiom = social_fol_two.replace('Q',Q).replace('R',R).replace('f',f)
                fol_axioms.append(social_fol_axiom)
            else:
                Q = data[i]['premise_switch']['0'][0]
                compare = ['more ','less ','better ','worse ']
                for comp in compare:
                    if comp in situation:
                        if comp+'than' in situation:
                            f = situation.split(comp)[0].split(' ')[-2]
                        else:
                            f = situation.split(comp)[1]
                g = data[i]['antonym_switch'][0]
                social_fol_axiom = social_fol_three.replace('Q',Q).replace('f',f).replace('g',g)
                fol_axioms.append(social_fol_axiom)
                
def temporal_fol():
    pass

def write_fol():
    with open('../data/FOL_axiom/fol_axiom.txt','a') as fol_file:
        for i, axiom in enumerate(fol_axioms):
            fol_file.write(axiom)
            if i != len(fol_axioms)-1:
                fol_file.write('\n')
        
if __name__ == "__main__":
    fol_axioms = []
    social_fol()
    write_fol()