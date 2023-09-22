from translation_metric import metric_text_to_smi, metric_fingerprint
import torch
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import csv
import os
from tqdm import tqdm
import argparse

device = torch.device("cuda")
torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='evalution',
                    help='train|evaluation')
args = parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, text_list, labels, tokenizer, max_length, mode):
       # self.dataloader = dataloader
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.text_list)
        #return self.dataloder.dataset

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        if self.mode == 'train':
            label_encoding = self.tokenizer(label, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')['input_ids'][0]
        else:
            label_encoding = label
        res = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_encoding
        }
        return res


class T5():
    def __init__(self):
        self.max_length = 256
        self.num_epochs = 10
        self.model = T5ForConditionalGeneration.from_pretrained('./t5_large').to(device)
        self.tokenizer = T5Tokenizer.from_pretrained("./t5_large", model_max_length=self.max_length)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-5)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_dir = './t5_model/t5_large_ChEBI'
        self.scheduler = None
        self.train_dataloader = None
        # self.eval_dataloader = None
        self.test_dataloader = None
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.x = []
        self.train_loss = []
        self.eval_loss = []
        self.eval_metric = []
        with open("GEM_data_translation_ChEBI_0804/train.source", "r", encoding="utf-8") as f:
            lines = f.readlines()
            texts = [x[6:].strip() for x in lines]
        self.texts = texts

        with open("GEM_data_translation_ChEBI_0804/train.target", "r", encoding="utf-8") as f:
            lines = f.readlines()
            labels = [x.strip() for x in lines]
        self.labels = labels

        with open("GEM_data_translation_ChEBI_0804/test.source", "r", encoding="utf-8") as f:
            lines = f.readlines()
            texts_test = [x[6:].strip() for x in lines]
        self.texts_test = texts_test
        # print('texts', self.texts_test[:1])

        with open("GEM_data_translation_ChEBI_0804/test.target", "r", encoding="utf-8") as f:
            lines = f.readlines()
            labels_test = [x.strip() for x in lines]
        self.labels_test = labels_test
        # print('labels', labels[:1])

        self.dataset = MyDataset(self.texts, self.labels, self.tokenizer, self.max_length, mode='train')
        self.dataset_test = MyDataset(self.texts_test, self.labels_test, self.tokenizer, self.max_length, mode='test')

    def data_loader(self):
        self.train_dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(self.dataset_test, batch_size=16, shuffle=True)

    def train(self):
        num_epochs = self.num_epochs
        num_training_steps = num_epochs * len(self.train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        self.model.train()
        for epoch in range(num_epochs):
            self.x.append(epoch)
            total_loss = 0.0
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # print('batch', batch)
                # input_ids = input_ids, attention_mask = attention_mask, labels = labels
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                # outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                # self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                total_loss += loss.item()
            avg_loss = total_loss/len(self.train_dataloader)
            self.train_loss.append(avg_loss)
            self.save_model(epoch)
            self.evalution(epoch)

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,  '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, 't5_large_' + str(epoch)))

    def evalution(self, epoch):
        print("*****this is evalution********")
        print('epoch', epoch)
        self.model.eval()
        total_loss = 0.0
        outputs = []
        labels = []
        for batch in self.test_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs_ids = self.model.generate(batch['input_ids'].to(self.device))
                # print('outputs_ids', outputs_ids)
                output = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
                # print('output:', output)
                outputs += output
                # print('label:', type(batch['labels']))
                labels += batch['labels']

            # self.metric.add_batch(predictions=predictions, references=batch['labels'])
        avg_loss = total_loss / len(self.test_dataloader)
        self.eval_loss.append(avg_loss)
        # self.eval_metric.append(metric.compute())
        # print(metric.compute())
        assert len(labels) == len(outputs)
        bleu_score, exact_match_score, levenshtein_score, validity_score1 = \
        metric_text_to_smi(labels, outputs)

        validity_score2, maccs_sims_score, rdk_sims_score, morgan_sims_score = \
        metric_fingerprint(labels, outputs)
        if epoch == self.num_epochs - 1:
            zipper = list(zip(labels, outputs))
            with open(os.path.join('output_csv_t5', f'pred_gold_predict_ChEBI.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['gold', 'predict'])
                writer.writerows(zipper)


if __name__ == '__main__':
    train = args.mode
    morgan_r = 2
    model_path = r''
    if train == 'train':
        run = T5()
        run.data_loader()
        run.train()
    elif 'evaluation':
        run = T5()
        checkpoints = []
        for i, checkpoint in enumerate(checkpoints):
            run.load_model(checkpoint)





# 配置训练参数
# training_args = TrainingArguments(
#     output_dir="./output_5e-5",  # 模型和日志的输出目录
#     learning_rate=5e-5,
#     overwrite_output_dir=True,  # 如果输出目录已存在，是否覆盖
#     num_train_epochs=10,  # 训练轮数
#     per_device_train_batch_size=8,  # 每个设备的训练批次大小
#     evaluation_strategy='epoch',
#     logging_dir="./logs",
#     logging_steps=3000,
# #     per_device_eval_batch_size=8,   # 每个设备的评估批次大小
#     save_steps=3000,  # 每多少步保存一次模型  #
#     save_total_limit=5,  # 最多保存多少个模型
# )

# # 定义Trainer对象
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=my_dataset,
#     eval_dataset=my_dataset_eval,
# )
#
# # 开始微调
# trainer.train()
#
# # 评估模型
# results = trainer.evaluate()
#
# # 保存微调后的模型
# model.save_pretrained("./fine-tuned-model")


