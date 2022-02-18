import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B_geo','I_geo','B_per','I_per','B_org','I_org'])



label_to_ids = {'B-geo': 1,
 'B-org': 2,
 'B-per': 3,
 'I-geo': 4,
 'I-org': 5,
 'I-per': 6,
 'O': 0}

ids_to_label = {1:'B-geo',
 2:'B-org',
 3:'B-per',
 4:'I-geo',
 5:'I-org',
 6:'I-per',
 0:'O'}


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        
        ## if sentence consist of more than 125 words, discard the later words.
        if(len(tokenized_sentence)>=125):
            return tokenized_sentence, labels
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels




class Ner_Data(Dataset):

    def __init__(self, data):
        self.data = data
        print("dataloader initialized")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         print(idx)
        sentence = self.data['text'][idx].strip().split()  
        word_labels = self.data['labels'][idx].split(" ") 

        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, tokenizer)
                
        sen_code = tokenizer.encode_plus(t_sen,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 128,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            )
             
            
        labels = [-100]*128
        for i, tok in enumerate(t_labl):
            if label_to_ids.get(tok) != None:
                labels[i+1]=label_to_ids.get(tok)

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] = torch.as_tensor(labels)

        return item


def get_batch_data(data):
    train_data = Ner_Data(data)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

    return train_dataloader