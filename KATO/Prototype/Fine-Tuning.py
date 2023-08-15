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

class FineTuner:
  def run(year = "2023", edit=False, calm_model="3b", sample_n = 100, n_token = 512):
    if edit:
      EDIT_STR = "_edit"
    else:
      EDIT_STR = ""
    
    TEACHER_DATA = pathlib.Path(__file__).parent / ("teacher_data_"+year+EDIT_STR+".csv")
    OUTPUT_DIR = pathlib.Path(__file__).parent / ("output_dir_"+calm_model+"_"+year+EDIT_STR)

    if sample_n > 0:
      list_train = pd.read_csv(TEACHER_DATA).sample(sample_n)
    else:
      list_train = pd.read_csv(TEACHER_DATA)

    MODEL_NAME="cyberagent/open-calm-"+calm_model  # modelを選択してください
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")

    def preprocess(examples, is_test=False):
      dataset = defaultdict(list)
    # original_sentenceは入力文章、generated_sentenceは出力してほしい文章を代入（入力文章部分も含む）
      for cnt in range(examples.shape[0]):
          example = examples.iloc[cnt]
          original_sentence, generated_sentence = example["original"], example["generated"]

          for i, sentence in enumerate(generated_sentence):
            input = tokenizer(original_sentence, max_length=n_token, padding="max_length", truncation=True)["input_ids"]
            labels = tokenizer(generated_sentence, max_length=n_token, padding="max_length", truncation=True)["input_ids"]

            if (not is_test) or (i == 0):
              dataset["input_ids"].append(input)
              dataset["labels"].append(labels)

      return dataset

    class Dataset(TorchDataset):
      def __init__(self, dataset, is_test=False):
        self.dataset = dataset
        self.is_test = is_test

      def __getitem__(self, idx):
        data = {'input_ids': torch.tensor(self.dataset["input_ids"][idx])}
        data['labels']=torch.tensor(self.dataset["labels"][idx])
        return data

      def __len__(self):
        return len(self.dataset["input_ids"])

    dataset_train = Dataset(preprocess(list_train))
    
    # LoRAのパラメータ
    lora_config = LoraConfig(
        r= 8, 
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # モデルの前処理
    model = prepare_model_for_int8_training(model)

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)

    training_config = TrainingArguments(
      output_dir = OUTPUT_DIR / "Learned",  # 出力したいディレクトリを入力してください
      num_train_epochs=3,
      learning_rate=3e-4,
      logging_steps=200,
      save_strategy="steps",
      save_steps=200,
      report_to="none",
      save_total_limit=3,
      push_to_hub=False,
      auto_find_batch_size=True,
      do_eval=False
    )
  
    trainer = Trainer(
        model = model,                         
        args = training_config,
        train_dataset = dataset_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    model.save_pretrained(OUTPUT_DIR / "peft") 
    
if __name__ == "__main__":
  FineTuner.run(year = "2015", edit=True, calm_model="3b", sample_n=30)
  