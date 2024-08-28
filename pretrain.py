from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import datasets
import torch
import numpy as np
from torch import nn



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # # Rouge expects a newline after each sentence
    # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
    #                   for pred in decoded_preds]
    # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
    #                   for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

tokenized_wow = DatasetDict.load_from_disk("wow_qg_large")

model_ckpt = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
tokenizer = T5Tokenizer.from_pretrained(model_ckpt)
question = "question:"
title = "title:"
context = "passage:"
eou = "<eou>"
tokenizer.add_tokens([question, title, context, eou], special_tokens=True)
tokenized_wow.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


batch_size = 8
model_dir = f"wow_pretrained_{model_ckpt}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=350,
    logging_strategy="steps",
    logging_steps=350,
    save_strategy="steps",
    save_steps=350,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    gradient_accumulation_steps=2,
    predict_with_generate=False,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

metric = datasets.load_metric("rouge")

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_wow["train"],
    eval_dataset=tokenized_wow["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{model_dir}/final")