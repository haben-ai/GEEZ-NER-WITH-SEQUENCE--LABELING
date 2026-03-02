from transformers import XLMRobertaForTokenClassification, AdamW
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label2id)).to(device)

train_dataset = GezeDataset(train_data)
test_dataset = GezeDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

# Training loop
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

epochs = 3
loss_fn = CrossEntropyLoss(ignore_index=-100)

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=False)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())
    scheduler.step()