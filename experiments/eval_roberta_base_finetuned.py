import numpy

def evaluation_by_category():
    category_folders = ['physical','social','material','temporal']
    num_of_entities = ["10","30","50","100","200"]
    # num_of_entities = ["200"]
    accuracy_dict = {}

    for num in num_of_entities:
        correct_count = 0
        mask_word = []
        result_output = []
        for category in category_folders:
            
            result_path = '../data/finetune_data/new_finetune/result/'+num+'_test_result.txt'
            with open(result_path, 'r') as result_file:
                result_list = result_file.readlines()
            # results = numpy.array_split(numpy.array(result_list),80)
            
            for index in range(20):
                
                # result_path = '../../data/finetune_data/new_finetune/result/'+num+'_test_result.txt'
                right_answer_path = '../data/finetune_data/new_finetune/'+category+"/"+num+'/test_sentences_m' + str(index) +'.txt'
                
                # with open(result_path, 'r') as result_file:
                #     result_list = result_file.readlines()
                # results = numpy.array_split(numpy.array(result_list),20)

                with open(right_answer_path, 'r') as answer_file:
                    right_answer = answer_file.readlines()
                    for answer in right_answer:

                        mask_word.append(answer.strip("\n"))

            
        
        for i in range(len(result_list)):
            first_answer = result_list[i].split('},')[0]
            
            if mask_word[i] in first_answer:
                answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"correct"
                result_output.append(answer_output)
                correct_count += 1
            else:
                answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"wrong"
                result_output.append(answer_output)
        
        output_file_path = 'result/'+num+'_output_correct_wrong.txt'
        with open(output_file_path,'w') as f:
            f.write('\n'.join(result_output))
            
            
        accuracy = correct_count/len(result_list)
        accuracy_dict[num] = accuracy
    
    return accuracy_dict

def evaluation_whole(num):
    
    correct_count = 0
    mask_word = []
    result_output = []
    
        
    result_path = '../data/finetune_data/axiom_w_perturbation/result/'+num+'_test_result.txt'
    with open(result_path, 'r') as result_file:
        result_list = result_file.readlines()
    # results = numpy.array_split(numpy.array(result_list),80)
    
    for index in range(20):
        
        # result_path = '../../data/finetune_data/new_finetune/result/'+num+'_test_result.txt'
        right_answer_path = '../data/finetune_data/axiom_w_perturbation/'+num+'/test_sentences_m.txt'
        
        # with open(result_path, 'r') as result_file:
        #     result_list = result_file.readlines()
        # results = numpy.array_split(numpy.array(result_list),20)

        with open(right_answer_path, 'r') as answer_file:
            right_answer = answer_file.readlines()
            for answer in right_answer:

                mask_word.append(answer.strip("\n"))

        
    
    for i in range(len(result_list)):
        first_answer = result_list[i].split('},')[0]
        
        if mask_word[i] in first_answer:
            answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"correct"
            result_output.append(answer_output)
            correct_count += 1
        else:
            answer_output = "right answer :"+ str(mask_word[i])+ "    result: "+ str(result_list[i])+"    score:"+"wrong"
            result_output.append(answer_output)
    print(len(result_output))
    output_file_path = 'result/axiom_w_perturbation/'+num+'_output_correct_wrong.txt'
    with open(output_file_path,'w') as f:
        f.write('\n'.join(result_output))
        
        
    accuracy = correct_count/len(result_list)
    print(accuracy)

if __name__ == "__main__":
    # accuracy = evaluation()
    # print(accuracy)
    evaluation_whole('30')


                        

