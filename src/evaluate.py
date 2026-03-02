from transformers import XLMRobertaForTokenClassification, AdamW
from torch.optim.lr_scheduler import StepLR

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        for p, l in zip(preds, labels):
            p = p[l!=-100].cpu().numpy()
            l = l[l!=-100].cpu().numpy()
            all_preds.extend(p)
            all_labels.extend(l)

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))]))