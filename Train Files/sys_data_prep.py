from transformers import T5ForConditionalGeneration, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm
import random
import torch


def crt(**kwargs):
    d = {}
    for k in kwargs.keys():
        d[k] = []
    return d
    
def apnd(d, i, **kwargs):
#     print(kwargs)
    for k in kwargs.keys():
#         print(type(k))
#         print(kwargs[k][i])
        d[k].append(kwargs[k][i])
    
    return d

def prep(df):
    out = crt(**df)
#     print(out)
    
    for i, jk in enumerate(df["gold_pass"]):
        if jk != float("-inf"):
            out = apnd(out, i, **df)
    
    return out

qg_wow = DatasetDict.load_from_disk("wow_qg_large")
qg_wowd =  qg_wow.map(prep, batched=True, remove_columns=qg_wow["train"].column_names)

query_tokenizer = T5Tokenizer.from_pretrained("wow_qg_t5-large/final")
# query_tokenizer.decode(qg_wowd["test"]["labels"][175])
qg_model = T5ForConditionalGeneration.from_pretrained("wow_qg_t5-large/final")
bs = 28

data_collator = DataCollatorForSeq2Seq(query_tokenizer)
qg_wowd.set_format(type='torch', columns=['input_ids', 'attention_mask'])
# qg_wowd.set_format(type='torch', columns=qg_wowd["train"].column_names)
train_loader = DataLoader(qg_wowd["train"], bs, shuffle=False, collate_fn=data_collator)
valid_loader = DataLoader(qg_wowd["valid"], bs, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(qg_wowd["test"], bs, shuffle=False, collate_fn=data_collator)

device = "cuda" if torch.cuda.is_available() else "cpu"
qg_model = qg_model.to(device)

mdf_dt = {"train": {"masked_response": []}, "valid": {"masked_response": []}, "test": {"masked_response": []}}
qg_model.eval()
with torch.no_grad():
    beam = random.choice([1, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 7])
    for i in tqdm(train_loader):
        tokes = qg_model.generate(input_ids=i["input_ids"].to(device), attention_mask=i["attention_mask"].to(device), num_beams=beam, max_length=128)
        texts = query_tokenizer.batch_decode(tokes)
        mdf_dt["train"]["masked_response"] += texts
        
    for i in tqdm(valid_loader):
        tokes = qg_model.generate(input_ids=i["input_ids"].to(device), attention_mask=i["attention_mask"].to(device), num_beams=beam, max_length=128)
        texts = query_tokenizer.batch_decode(tokes)
        mdf_dt["valid"]["masked_response"] += texts
        
    for i in tqdm(test_loader):
        tokes = qg_model.generate(input_ids=i["input_ids"].to(device), attention_mask=i["attention_mask"].to(device), num_beams=beam, max_length=128)
        texts = query_tokenizer.batch_decode(tokes)
        mdf_dt["test"]["masked_response"] += texts

        
dft = DatasetDict()
dft["train"] = Dataset.from_dict(mdf_dt["train"])
dft["valid"] = Dataset.from_dict(mdf_dt["valid"])
dft["test"] = Dataset.from_dict(mdf_dt["test"])
dft.save_to_disk("generated_data")
        