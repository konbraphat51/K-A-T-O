import pandas as pd

TEACHER_DATA = "teacher_data_2015.csv"
SAMPLE_N = 50

list_train = pd.read_csv(TEACHER_DATA).sample(SAMPLE_N)

#https://qiita.com/Mizuiro__sakura/items/058d2590d31e9f8aeeaa

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL_NAME="cyberagent/open-calm-small"  # modelを選択してください
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForCausalLM.from_pretrained(MODEL_NAME)

n_token = 512
from collections import defaultdict
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

from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
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

from transformers import Trainer, TrainingArguments
training_config = TrainingArguments(
  output_dir = 'output_dir',  # 出力したいディレクトリを入力してください
  num_train_epochs = 4, 
  per_device_train_batch_size = 8,
  per_device_eval_batch_size = 8,
  warmup_steps = 100,
  weight_decay = 0.1,
  save_steps = 10000  # 用いるdatasetに応じてsave_stepsは変えてください
)

trainer = Trainer(
    model = model,                         
    args = training_config,
    train_dataset = dataset_train,
    
)

trainer.train()

model.save_model("prototype_2015")