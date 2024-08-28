from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5ForConditionalGeneration
import torch
from transformers import T5Tokenizer

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

beam = 6
top_p = 1
top_k = 50
bg = 3


def prep1(df):
    out = {"pass_label": [], "attention_mask": [], "input_ids": []}
    inp = torch.tensor(df["input_ids"]).to(device)
    att = torch.tensor(df["attention_mask"][i]).to(device)
    query_tokenizer.decode(qg_model.generate(input_ids=inp, attention_mask=att, num_beams=beam, min_length = 8, max_length=64, no_repeat_ngram_size=3, do_sample=False, top_p=top_p, top_k=top_k, num_beam_groups=bg, diversity_penalty=0.5)[:, 1:-1])
    for i, _ in enumerate(df["gold_pass"]):
        # print(type(k))
        if df["gold_pass"][i] == float("-inf"):
            continue
        inp = torch.tensor(df["input_ids"][i]).view(1, -1).to(device)
        att = torch.tensor(df["attention_mask"][i]).view(1, -1).to(device)
        input_q = df["last_ut"][i] + " <eou> " + query_tokenizer.decode(qg_model.generate(input_ids=inp, attention_mask=att, num_beams=beam, min_length = 8, max_length=64, no_repeat_ngram_size=3, do_sample=False, top_p=top_p, top_k=top_k, num_beam_groups=bg, diversity_penalty=0.5).view(-1)[1:]).replace("<nok>", "")
        input_ids = []
        attention_mask = []
#         out["pass_label"].append(int(df["gold_pass"][i]))
        for j, p in enumerate(df["all_pass"][i]):

            if j == int(df["gold_pass"][i]):
                # pass_label.append(1)
                p = (" ".join(df["all_sen"][i]))

            p = p.replace("no_passages_used", "")
            t = df["all_topic"][i][j]
            inp = tokenizer(f"question: {input_q} title: {t} passage: {p}", max_length=512, truncation=True, padding=False)
            out["input_ids"].append(inp["input_ids"])
            out["attention_mask"].append(inp["attention_mask"])
            
            out["pass_label"].append(int(df["gold_pass"][i]))
#             out["labels"].append(tokenizer(gpe_out)["input_ids"])
#         out["input_ids"].append(torch.tensor(input_ids))
#         out["attention_mask"].append(torch.tensor(attention_mask))
        # out["masked_response"].append(input_q)
        # out["resp"].append(df["response"][i])

    return out

qg_wow = DatasetDict.load_from_disk("wow_qg_large")
rank_wow = DatasetDict.load_from_disk("wow_rank_kel_base")

qg_wowd =  qg_wow.map(prep, batched=True, remove_columns=qg_wow["train"].column_names, batch_size=28)


from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("wow_rank_t5-base/final")
question = "question:"
title = "title:"
context = "context:"
eou = "<eou>"
tokenizer.add_tokens([question, title, context, eou], special_tokens=True)



query_tokenizer = T5Tokenizer.from_pretrained("wow_qg_t5-large/final")
# query_tokenizer.decode(qg_wowd["test"]["labels"][175])
qg_model = T5ForConditionalGeneration.from_pretrained("wow_qg_t5-large/final")

device = "cuda" if torch.cuda.is_available() else "cpu"
qg_model = qg_model.to(device)

rank_data = qg_wowd["test"].map(prep1, batched=True, remove_columns=qg_wowd["test"].column_names)
rank_data.save_to_disk(f"wow_qg_test_data_beam{beam}p{top_p}k{top_k}bg{bg}")
