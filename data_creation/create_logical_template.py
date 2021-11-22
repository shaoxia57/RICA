import json

def load_file(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.read())
    
    return data

def temporal_after_template(data):
    template_list = []
    for element in data.values():
        
        cause = element['cause']
        results = element['result']
        
        if 'ing' not in cause:
            if ' ' in cause:
                # after_ing = " ".join(cause.split(' ')[1:])
                cause = str(cause.split(' ')[0])+'ing '+" ".join(cause.split(' ')[1:])
            else:
                cause = cause+"ing"
        
        for result in results:
            if 'ing' not in result:
                if ' ' in result:
                    # after_ing = " ".join(result.split(' ')[1:])
                    result = str(result.split(' ')[0])+'ing '+" ".join(result.split(' ')[1:])
                else:
                    result = result+'ing'
            template = 'A was '+cause+', so A was '+result+' after A was '+cause+'.'
            template_list.append(template)
    
    return template_list

def temporal_before_template(data):
    template_list = []
    for element in data.values():
        
        causes = element['cause']
        result = element['result']
        
        if 'ing' not in result:
            if ' ' in result:
                # after_ing = " ".join(cause.split(' ')[1:])
                result = str(result.split(' ')[0])+'ing '+" ".join(result.split(' ')[1:])
            else:
                result = result+"ing"
        
        for cause in causes:
            if 'ing' not in cause:
                if ' ' in cause:
                    # after_ing = " ".join(result.split(' ')[1:])
                    cause = str(cause.split(' ')[0])+'ing '+" ".join(cause.split(' ')[1:])
                else:
                    cause = cause+'ing'
            template = 'A was '+result+', so A was '+cause+' before A was '+result+'.'
            template_list.append(template)
    
    return template_list

if __name__ == "__main__":
    after_data = load_file('../data/temporal_example_after.txt')
    template_after = temporal_after_template(after_data)

    before_data = load_file('../data/temporal_example_before.txt')
    template_before = temporal_before_template(before_data)

    with open('../data/scratch/temporal_sentences_after.txt','w') as f:
        f.write('\n'.join(template_after))
    with open('../data/scratch/temporal_sentences_before.txt','w') as f:
        f.write('\n'.join(template_before))
