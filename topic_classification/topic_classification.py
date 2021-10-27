# 2021.09.22
# @Jihyun22

import transformers
print(transformers.__version__)

get_ipython().system('huggingface-cli login')
get_ipython().system('pip install hf-lfs')
get_ipython().system('git config --global user.email eliza.dukim@gmail.com')
get_ipython().system('git config --global user.name KimDaeUng')

task = "ynat"
model_checkpoint = "klue/bert-base"
batch_size = 256


from datasets import load_dataset
dataset = load_dataset('klue', 'ynat') # klue 의 task=nil 을 load
dataset # dataset 구조 확인

dataset['train'][0]


import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(dataset["train"])


import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['title'], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

dataset['train'].features['label'].num_classes

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
num_labels = 7 
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                           num_labels=num_labels)


import os

model_name = model_checkpoint.split("/")[-1]
output_dir = os.path.join("test-klue", "ynat") # task 별로 바꿔줄 것
logging_dir = os.path.join(output_dir, 'logs')
args = TrainingArguments(
    # checkpoint, 모델의 checkpoint 가 저장되는 위치
    output_dir=output_dir,
    overwrite_output_dir=True,

    # Model Save & Load
    save_strategy = "epoch", # 'steps'
    load_best_model_at_end=True,
    save_steps = 500,


    # Dataset, epoch 와 batch_size 선언
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    
    # Optimizer
    learning_rate=2e-5, # 5e-5
    weight_decay=0.01,  # 0
    # warmup_steps=200,

    # Resularization
    # max_grad_norm = 1.0,
    # label_smoothing_factor=0.1,


    # Evaluation 
    metric_for_best_model='eval_f1', # task 별 평가지표 변경
    evaluation_strategy = "epoch",

    # HuggingFace Hub Upload, 모델 포팅을 위한 인자
    push_to_hub=True,
    push_to_hub_model_id=f"{model_name}-finetuned-{task}",

    # Logging, log 기록을 살펴볼 위치, 본 노트북에서는 wandb 를 이용함
    logging_dir=logging_dir,
    report_to='wandb',

    # Randomness, 재현성을 위한 rs 설정
    seed=42,
)

from datasets import list_metrics, load_metric
metrics_list = list_metrics()
len(metrics_list)
print(', '.join(metric for metric in metrics_list))

metric_macrof1 = load_metric('f1')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return metric_macrof1.compute(predictions=predictions,
                                  references=labels, average='macro')
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


import wandb
wandb.login()
id = wandb.util.generate_id()
print(id)

wandb.init(project='klue', # 실험기록을 관리한 프로젝트 이름
           entity='dukim', # 사용자명 또는 팀 이름
           id='3bso6955',  # 실험에 부여된 고유 아이디
           name='ynat',    # 실험에 부여한 이름               
          )

trainer.train()
wandb.finish()
trainer.evaluate()
trainer.push_to_hub()

from transformers import AutoModelForSequenceClassification
trainer.save('/test-klue/ynat/model.h5')

get_ipython().system(' pip install optuna')
get_ipython().system(' pip install ray[tune]')
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
trainer_hps = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"].shard(index=1, num_shards=10), 
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


wandb.init()
wandb.login()
id = wandb.util.generate_id()
print(id)

best_run = trainer_hps.hyperparameter_search(n_trials=5, direction="maximize")

best_run

