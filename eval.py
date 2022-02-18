import torch
from data import device
from data_loader import ids_to_label
# from model import model2

model2 = torch.load('saved_model\model.pt')

def valid(testing_loader):
    # model2.eval()
    eval_preds, eval_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            preds= model2(input_ids=ids, attention_mask=mask, labels=labels)

            eval_logits = preds['logits'] 
                          
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model2.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            

    labels = [ids_to_label[id.item()] for id in eval_labels]
    predictions = [ids_to_label[id.item()] for id in eval_preds]

    return labels, predictions


