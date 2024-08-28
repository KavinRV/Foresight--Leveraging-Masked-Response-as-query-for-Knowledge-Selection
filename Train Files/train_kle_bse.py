from datasets import Dataset, DatasetDict
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import datasets
from typing import Optional
from transformers.trainer import is_datasets_available, seed_worker

data_dir = "wow_gmr_0to50"
tokenized_wow = DatasetDict.load_from_disk(data_dir)
tokenized_wow.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "pass_label"])
print(data_dir)


class RankT5GPE(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        config.rank_score_index = 32019
        config.n_pass = 7
        config.output_hidden_states = True
        super().__init__(config)
        self.rank_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rank_id = config.rank_score_index
        self.n_pass = config.n_pass

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)

        # input_ids = input_ids.view(batch_size*n_pass, -1)
        # attention_mask = attention_mask.view(batch_size*n_pass, -1)
        
        
        
        if labels != None and decoder_input_ids == None:
#             batch_size, decoder_seq_len = labels.size()
#             labels = labels.view(batch_size, 1, decoder_seq_len).contiguous()
#             labels = labels.expand(batch_size, n_pass, decoder_seq_len).contiguous()

#             labels = labels.view(batch_size*n_pass, -1)
            decoder_input_ids = self._shift_right(labels)
#             print(decoder_input_ids.size())
# <pad> k1 k2 k3 
#       -1 -2 4 --> 1
#            rs --> 2

        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, **kwargs)
        rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :]) # bn * dsl * 512 --> bn * 512 --> bn * v 
        out.rank_score = rank_score[:, self.rank_id].view(-1, self.n_pass) # bn * v --> bn --> b * n


        if labels != None:
            logits = out.logits
            batch_size_n, sequence_length, vocab_size = logits.size()

            logits_flat = logits.view(batch_size_n * sequence_length, vocab_size)
            labels_flat = labels.view(-1)
            mask = (labels_flat != -100)
            arry = torch.arange(batch_size_n * sequence_length).to(logits_flat.device)

            selected_logits = logits_flat[arry[mask], labels_flat[mask]]
            output_logits = torch.full((batch_size_n * sequence_length,), 0, dtype=logits.dtype, device=logits.device)
            output_logits[mask] = selected_logits

            # output_logits = nn.functional.softmax(output_logits, dim=-1)
            output_logits = output_logits.view(batch_size_n, -1).sum(-1)
            # output_logits = output_logits.view(batch_size_n, -1).prod(-1)
            out.gpe_score = output_logits.view(int(batch_size_n/self.n_pass), self.n_pass)

        else:
            out.gpe_score = None


        if pass_label != None:
            pass_label = pass_label[::self.n_pass] # bn
            # 3 --> log_softmax(rank_score)[3]
            # Q -> [23, 34, 48, 32] pl -> [2] rs -> [-2, -3, 5, -1] -> [_, _, __, _]
            # Q -> [23, 48, 34, 32] pl -> [2] rs -> [-2, 5, -3, -1] -> [_, _, __, _]
            rank_score = out.rank_score # [-2, 5, -3, -1] 
            gen_score = out.gpe_score # [-3, 4, -2, -1]

            loss_fct1 = nn.CrossEntropyLoss()
            loss_fct2 = nn.CrossEntropyLoss()

            rank_loss = loss_fct1(rank_score, pass_label.view(-1))
            gen_loss = loss_fct2(gen_score, pass_label.view(-1))

            loss = rank_loss + gen_loss
            out.loss = loss

        return out
    
       
mod_ckp = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(mod_ckp)
config = T5Config.from_pretrained(mod_ckp)
config.output_hidden_states = True
config.rank_score_index = tokenizer.convert_tokens_to_ids("<extra_id_80>")
config.n_pass = 7
model = RankT5GPE.from_pretrained(mod_ckp, config=config)


batch_size = 3*model.config.n_pass 
model_dir = f"wow_rank_{mod_ckp}_0-50"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=250,
    logging_strategy="steps",
    logging_steps=250,
    save_strategy="steps",
    save_steps=500,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    gradient_accumulation_steps=12
)


data_collator = DataCollatorForSeq2Seq(tokenizer)


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
#         print(inputs.get("input_ids").size())
        outputs = model(**inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, shuffle=False, **dataloader_params))
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, shuffle=False, **dataloader_params))

trainer = CustomTrainer(
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

