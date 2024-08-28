import torch.nn.functional as F

from datasets import DatasetDict
from datasets import Dataset as DS
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import datasets
from typing import Optional
from transformers.trainer import is_datasets_available, seed_worker
from transformers.modeling_outputs import Seq2SeqLMOutput

tokenized_wow = DatasetDict.load_from_disk("wow_gmr_0to50")
tokenized_wow.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "pass_label"])
# tok_valid = 


class RankT5GPE(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
#         config.rank_score_index = 32019
#         config.n_pass = 7
#         config.output_hidden_states = True
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


        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, **kwargs)
        rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
#         try:
        out.rank_score = rank_score[:, self.rank_id].view(-1, self.n_pass)
#         except RuntimeError:
#             print(rank_score.size())
#             assert 2 == 3
            

        loss = None

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

            output_logits = output_logits.view(batch_size_n, -1).sum(-1)
            out.gpe_score = output_logits.view(int(batch_size_n/self.n_pass), self.n_pass)

        else:
            out.gpe_score = None
            
#         out.rank_loss = None
#         out.kl_d_loss = None


        if pass_label != None:
            pass_label = pass_label[::self.n_pass]
            rank_score = F.log_softmax(out.rank_score/float(config.temp2), dim=-1) # S
            gen_score = F.log_softmax((out.gpe_score/float(config.temp1)), dim=-1) # T
            # K(rs||g) = g.log(rs/g)

            loss_fct1 = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_fct3 = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_fct2 = nn.CrossEntropyLoss()

            rank_loss = loss_fct2(out.rank_score, pass_label.view(-1))
            kl_d_loss = loss_fct1(rank_score, gen_score.detach()) + loss_fct3(gen_score, rank_score.detach())
            # kl = softmax(gen_score).log(softmax(rank_score)/softmax(gen_score))
            
#             out.rank_loss = rank_loss
#             out.kl_d_loss = kl_d_loss
            loss = rank_loss + config.lamda*kl_d_loss

        ret =  Seq2SeqLMOutput(
            loss=loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            decoder_hidden_states=out.decoder_hidden_states,
            decoder_attentions=out.decoder_attentions,
            cross_attentions=out.cross_attentions,
            encoder_last_hidden_state=out.encoder_last_hidden_state,
            encoder_hidden_states=out.encoder_hidden_states,
            encoder_attentions=out.encoder_attentions,
        )
        ret.rank_score = out.rank_score
        return ret
    
       
mod_ckp = "wow_pretrained_t5-base/final"
tokenizer = T5Tokenizer.from_pretrained(mod_ckp)
# question = "question:"
# title = "title:"
# context = "context:"
# eou = "<eou>"
# tokenizer.add_tokens([question, title, context, eou], special_tokens=True)
config = T5Config.from_pretrained(mod_ckp)
config.output_hidden_states = True
config.rank_score_index = tokenizer.convert_tokens_to_ids("<extra_id_80>")
config.n_pass = 7
config.temp1 = 1
config.temp2 = 1
config.lamda = 0.5
model = RankT5GPE.from_pretrained(mod_ckp, config=config)

batch_size = 3*model.config.n_pass 
# model = torch.nn.DataParallel(model)
# t = 2
model_dir = f"wow_rank_pretrained_sym_kld_tem:T{model.config.temp1}S{model.config.temp2}"

args = Seq2SeqTrainingArguments(
    model_dir,
    label_names=["pass_label", "labels"],
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
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    gradient_accumulation_steps=12
)


data_collator = DataCollatorForSeq2Seq(tokenizer)


class CustomTrainer(Seq2SeqTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
# #         print(inputs.get("input_ids").size())
#         outputs = model(**inputs)

#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss
    
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
    trainer.train(f"{model_dir}/checkpoint-3500")
    trainer.save_model(f"{model_dir}/final")