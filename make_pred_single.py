import pandas as pd 
import torch
from torch.utils.data import Dataset
# from model import model2

from data import device, tokenizer
from data_loader import ids_to_label
model2 = torch.load('saved_model\model',map_location=torch.device('cpu'))

class process_sentence_single(Dataset):

    def __init__(self, text):
        self.text = text
        # print("dataloader initialized")
        
    def __len__(self):
        return 1

    def __getitem__(self,idx):

        sentence = self.text.strip().split() 
        
        tokenized_sentence = []

        for word in sentence:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
        
        
        sen_code = tokenizer.encode_plus(tokenized_sentence,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            return_attention_mask = True,  # Generate the attention mask
            )

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        return item



def make_single_pred(sentence):

    # get the processed input_ids and mask
    # test_text = "Mark is the ceo of Facebook. located in California ."
    test_text = sentence
    pre_text = process_sentence_single(test_text)
    text= pre_text[0]

    ids = text ['input_ids']
    mask = text ['attention_mask']

    
    #make prediction
    
    test_pred = model2(input_ids=torch.unsqueeze(ids,0).to(device), attention_mask=torch.unsqueeze(mask,0).to(device))

    
    ## flatten prediction
    active_logits = test_pred[0].view(-1, model2.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
#     print("\nFlatten Predictions.....\n")
#     print(flattened_predictions)

    
    
#     print("\n printing tokens.....")
#     for i in torch.unsqueeze(ids,0):
#         print(tokenizer.convert_ids_to_tokens(i))

    # convert ids to corresponding tokens
    text_tokens= tokenizer.convert_ids_to_tokens(ids)

    # convert predctions to labels
    text_labels = []
    for i in flattened_predictions.squeeze(0).cpu().numpy():
        text_labels.append(ids_to_label.get(i))

#     print("\n printing predicted token labels.....")
#     print(text_labels)

    # remove first and last tokens ([CLS] and [SEP])
    text_tokens = text_tokens[1:-1]
    text_labels = text_labels[1:-1]


#     print("\n printing tokens with labels")
#     print(text_tokens)
#     print(text_labels)
    
    return text_tokens, text_labels



# df

def extract_entity(test_file):
    df = pd.DataFrame(columns = ['Free flow of Text','Extracted Name','Extracted Location','Extracted Organization'])

    for sent in test_file:
    #     print(sent)
        text_sen = sent
        per=[]
        geo=[]
        org=[]
        txt, lbl = make_single_pred(sent)
        for text, label in zip(txt,lbl):
    #         print(text,label)
            
            if(label == 'I-per'):
                if not text.startswith('##'):
    #                 print("####")
    #                 print(text)
                    if(len(per)<=0):
                        per.append(text)
                    else:
                        per[-1] = per[-1]+' '+ text
                    continue

            if(label[2:] == 'per'):
                if text.startswith('##'):
                    per[-1] = per[-1]+text[2:]
                else:
                    per.append(text)

                    
                    
            if(label == 'I-geo'):
                
                if not text.startswith('##'):
    #                 print("####")
    #                 print(text)
                    if(len(geo)<=0):
                        geo.append(text)
                    else:
                        geo[-1] = geo[-1]+' '+ text
                    
                    continue
                    
            if(label[2:] == 'geo'):
                if text.startswith('##'):
                    geo[-1] = geo[-1]+text[2:]
                else:
                    geo.append(text)

                    
                    
            if(label == 'I-org'):
                if not text.startswith('##'):
    #                 print("####")
    #                 print(text)
                    if(len(org)<=0):
                        org.append(text)
                    else:
                        org[-1] = org[-1]+' '+ text
                    continue
                    
            if(label[2:] == 'org'):
                if text.startswith('##'):
                    org[-1] = org[-1]+text[2:]
                else:
                    org.append(text)
                    
    #     df.append({'Free flow of Text':text_sen, 'Extracted Name':per, 'Extracted Location':geo,'Extracted Organization':org}, ignore_index=True)
            
        new_record = pd.DataFrame([[text_sen,per,geo,org]], columns = ['Free flow of Text','Extracted Name','Extracted Location','Extracted Organization'])

        df = pd.concat([df, new_record])
        
    df.reset_index(drop=True, inplace=True)

    return df
