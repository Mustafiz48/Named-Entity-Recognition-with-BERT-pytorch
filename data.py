
import torch
import pandas as pd
from data_loader import get_batch_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B_geo','I_geo','B_per','I_per','B_org','I_org'])



def remove_extra_tags(data): 
    '''
        Here we will remove all the tags except for 'Org','Geo', and 'Per'
        type. 
    '''
#     data['labels'] = data['labels'].str.replace('-','_')
    data['labels'] = data['labels'].str.replace('B_gpe','O')
    data['labels'] = data['labels'].str.replace('I_gpe','O')
    data['labels'] = data['labels'].str.replace('B_tim','O')
    data['labels'] = data['labels'].str.replace('I_tim','O')
    data['labels'] = data['labels'].str.replace('B_eve','O')
    data['labels'] = data['labels'].str.replace('I_eve','O')
    data['labels'] = data['labels'].str.replace('B_nat','O')
    data['labels'] = data['labels'].str.replace('I_nat','O')
    data['labels'] = data['labels'].str.replace('B_art','O')
    data['labels'] = data['labels'].str.replace('I_art','O')
    
    return data


def get_data(data):
    data = remove_extra_tags(data)

    '''
    This is to remove the sentences that doesn't contain our targeted entities.
    '''
    sum=0
    for index, i in enumerate(data['labels']):
        a=set(i.split(' '))
        if(len(a)<=1):
            data.drop(labels=index, axis=0,inplace=True)
            sum+=1    
    print(sum)
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    return data

data =pd.read_csv('input/ner.csv')

data = get_data(data)

print("\n printing data sample.......\n")
print(data.head)

train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

train_dataloader = get_batch_data(train_dataset)
test_data_loader = get_batch_data(test_dataset)