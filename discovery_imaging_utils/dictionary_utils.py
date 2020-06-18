import os
import glob
import json
import numpy as np




def json_path_to_dict(path_to_json_file):
    
    with open(path_to_json_file,'r') as temp_file:
        
        json_contents = temp_file.read()
        dict_object = json.loads(json_contents)
        
    return dict_object


def save_dictionary(dictionary, path_for_dictionary_dir, overwrite = False):
    
        
    if path_for_dictionary_dir[-5:] == '.json':
        
        with open(os.path.join(path_for_dictionary_dir), 'w') as temp_file:
            json_dict = json.dumps(dictionary, indent = 4, sort_keys = True)
            temp_file.write(json_dict)        

    else:
        
        if os.path.exists(path_for_dictionary_dir) == True:
        
            if overwrite == False:

                raise NameError('Error: Parc Timeseries Dictionary Directory Already Exists at this path')

        else:
        
            os.makedirs(path_for_dictionary_dir)
        
        
        
        for temp_key in dictionary.keys():


            if type(dictionary[temp_key]) == dict:

                to_overwrite = overwrite
                save_dictionary(dictionary[temp_key], os.path.join(path_for_dictionary_dir, temp_key), overwrite = to_overwrite)

            else:


                if path_for_dictionary_dir[-3:] == 'txt':

                    with open(os.path.join(path_for_dictionary_dir, temp_key), 'w') as temp_file:
                        temp_file.write(str(dictionary[temp_key]))

                else:

                    np.save(os.path.join(path_for_dictionary_dir, temp_key), dictionary[temp_key])
                 
            
    return
                
                
    
    
    
def load_dictionary(dictionary_dir_path):
    
    dictionary = {}
    
    os.chdir(dictionary_dir_path)
    directory_contents = os.listdir()
    directory_final_name = dictionary_dir_path.split('/')[-1]
    for temp_file in directory_contents:
        
        if os.path.isdir(temp_file):
            
            dictionary[temp_file] = load_dictionary(os.path.join(dictionary_dir_path, temp_file))
            os.chdir(dictionary_dir_path)
            
        else:
            
            if temp_file[-5:] == '.json':
                                
                with open(temp_file, 'r') as temp_json:
                    json_contents = temp_json.read()
                    dictionary[temp_file] = json.loads(json_contents)
                    
            else:
            
                if dictionary_dir_path[-3:] == 'txt':

                    with open(temp_file,'r') as temp_reading:
                        file_contents = temp_reading.read()

                        try:
                            dictionary[temp_file.split('.')[0]] = float(file_contents)
                        except:

                            if file_contents[0] == '[' and file_contents[-1] == ']':
                                split_contents = file_contents[1:-1].split(',')
                                dictionary[temp_file.split('.')[0]] = []
                                for temp_item in split_contents:
                                    split_limited_1 = temp_item.replace(' ','')
                                    split_limited_2 = split_limited_1.replace("'","")
                                    dictionary[temp_file.split('.')[0]].append(split_limited_2)

                            else:
                                dictionary[temp_file.split('.')[0]] = file_contents                

                else:

                    dictionary[temp_file.split('.')[0]] = np.load(temp_file)
                
                
                
    return dictionary


def flatten_dictionary(dictionary):
    
    
    def inner_function(sub_dict, name_beginning):
        
        inner_dict = {}
        
        for temp_key in sub_dict.keys():
            
            if name_beginning != '':
                new_name_beginning = name_beginning + '_' + temp_key
            else:
                new_name_beginning = temp_key
                
            if type(sub_dict[temp_key]) == dict:
                
                new_dictionary = inner_function(sub_dict[temp_key], new_name_beginning)
                for temp_inner_key in new_dictionary.keys():
                    inner_dict[temp_inner_key] = new_dictionary[temp_inner_key]
                
            else:
                inner_dict[new_name_beginning] = sub_dict[temp_key]
                
        return inner_dict
    
    flattened_dictionary = inner_function(dictionary, '')
    return flattened_dictionary
