#https://qiita.com/Mizuiro__sakura/items/058d2590d31e9f8aeeaa
#https://note.com/npaka/n/na5b8e6f749ce

import pandas as pd
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import datetime
import json

class FineTuner:
    def run(self, finetuner_properties):
        self.finetuner_properties = finetuner_properties
        df_teacher = self.get_teacher_data(finetuner_properties.year, finetuner_properties.sample_n)
        self.train_model(self, df_teacher)

    def train_model(self, df_teacher):
        MODEL_NAME=self.get_original_model_name()
        
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME, load_in_8bit=self.finetuner_properties.useint8, device_map="auto")
        model=AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=self.finetuner_properties.useint8, device_map="auto")

        def preprocess(examples, is_test=False):
            dataset = defaultdict(list)
            for cnt in range(examples.shape[0]):
                example = examples.iloc[cnt]
                generated_sentence = example[self.finetuner_properties.text_row]

                for i, sentence in enumerate(generated_sentence):
                    output = tokenizer(generated_sentence, max_length=self.finetuner_properties.n_token, padding="max_length", truncation=True)["input_ids"]
                    if (not is_test) or (i == 0):
                       dataset["input_ids"].append(output)

            return dataset

        class Dataset(TorchDataset):
            def __init__(self, dataset, is_test=False):
                self.dataset = dataset
                self.is_test = is_test

            def __getitem__(self, idx):
                data = {'input_ids': torch.tensor(self.dataset["input_ids"][idx])}
                #data['labels']=torch.tensor(self.dataset["labels"][idx])
                return data

            def __len__(self):
                return len(self.dataset["input_ids"])

        dataset_train = Dataset(preprocess(df_teacher))

        if self.finetuner_properties.useint8:
            model = prepare_model_for_int8_training(model)

        # LoRAモデルの準備
        model = get_peft_model(model, self.finetuner_properties.lora_config)
    
        self.finetuner_properties.training_config.output_dir = self.get_output_dir() / "Learned"
    
        trainer = Trainer(
            model = model,                         
            args = self.finetuner_properties.training_config,
            train_dataset = dataset_train,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        model.config.use_cache = False
        trainer.train()
        model.config.use_cache = True

        model.save_pretrained(self.get_output_dir() / "peft") 

    def get_teacher_data(self):
        '''
        教師データ取り込み
        '''
        data_path = self.get_teacher_data_path(self.finetuner_properties.year)
        if self.finetuner_properties.sample_n > 0:
            return pd.read_csv(data_path).sample(self.finetuner_properties.sample_n)
        else:
            return pd.read_csv(data_path)

    def get_teacher_data_path(self):
        return pathlib.Path(__file__).parent / ("teacher_data_"+self.finetuner_properties.year+".csv")

    def get_output_dir(self):
        return pathlib.Path(__file__).parent / ("output_dir_"+ self.finetuner_properties.id)
    
    def get_original_model_name(self):
        raise NotImplementedError()
    
class FineTunerProperties:
    def __init__(
        self,
        year:str = "2023",
        sample_n:int = 100,
        text_row:str = "text",
        n_token = None,
        useint8:bool = True,
        lora_r:int = 8,
        lora_alpha:int = 16,
        lora_dropout:float = 0.05,
        lora_bias:str = "none",
        ta_epochs:int = 4,
        ta_logging_steps:int = 200,
        ta_save_steps:int = 100000,
        ta_save_total_limit:int = 3,
        ta_train_batch_size:int = 8,
        ta_warmup_steps:int = 200,
        ta_weight_decay:float = 0.1,
        ta_learning_rate:float = 5e-4
    ):
        self.args = locals()
        
        self.initialize_id()
        self.year = year
        self.sample_n = sample_n
        self.text_row = text_row
        self.n_token = n_token
        self.useint8 = useint8
        
        # LoRAのパラメータ
        self.lora_config = LoraConfig(
            r= lora_r, 
            lora_alpha=lora_alpha,
            target_modules=["query_key_value"],
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        self.training_config = TrainingArguments(
            num_train_epochs=ta_epochs,
            logging_steps=ta_logging_steps,
            save_strategy="steps",
            save_steps=ta_save_steps,
            report_to="none",
            save_total_limit=ta_save_total_limit,
            push_to_hub=False,
            do_eval=False,
            per_device_train_batch_size = ta_train_batch_size,
            warmup_steps = ta_warmup_steps,
            weight_decay = ta_weight_decay,
            learning_rate=ta_learning_rate
        )
        
    def initialize_id(self):
        '''
        今回の実験IDを作る
        '''
        current_time = datetime.datetime.now()
        #YYYYMMDDhhmm
        time_str = current_time.strftime("%Y%m%d%H%M")
        self.id = time_str
        
    def save(self):
        '''
        プロパティを保存する
        '''
        json_data = json.dumps(self.args)
        with open(self.get_output_dir() / "properties.json", mode='w') as f:
            f.write(json_data)
        
    def get_output_dir(self):
        return pathlib.Path(__file__).parent / ("output_dir_"+ self.id)    
            
if __name__ == "__main__":
  FineTuner.run(year = "2015", edit=False, calm_model="1b", sample_n=100, useint8=True)
  