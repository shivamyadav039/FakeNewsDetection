# download_bert_model.py
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')

print("BERT model and tokenizer downloaded and saved to ./bert_model")
