import torch
from model import get_model
from data  import device

learning_rate = 0.0001
batch_size = 64
epochs = 5

model2 = get_model()

optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)


def train_loop(train_dataloader, optimizer):
    size = len(train_dataloader.dataset)
    train_loss =0
    for i,sample in enumerate(train_dataloader):
        optimizer.zero_grad()
#         print(sample)
        ids=sample['input_ids'].to(device)
        mask=sample['attention_mask'].to(device)
        labels = sample['labels'].to(device)
        pred = model2(input_ids=ids, attention_mask=mask ,labels = labels )
        loss = pred[0]
        
#         print(f"loss: {loss.item()}")
        train_loss+=loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        
        if(i>0 and i % 500==0):
            print(f"loss: {train_loss/i:>4f}  [{i:>5d}/{size/32}]")
    return train_loss



def train(train_dataloader):
    epochs = 5
    train_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(train_dataloader, optimizer)
        train_loss.append(loss)
    #     test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model2, 'saved_model/')


