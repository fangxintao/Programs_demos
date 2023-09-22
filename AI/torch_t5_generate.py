from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv


def generate_text(samples):
    # input_text = "Translate the following English text to French: 'Hello, how are you?'"
    # 编码输入文本
    # input_ids = tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

    # 使用模型生成文i本
   # print('samples:', samples)
    output_ids = model.generate(samples['input_ids'], num_beams=1, max_new_tokens=512, do_sample=False)
    #print('output_ids:', output_ids)

    # 解码生成的文本
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #print('outputs', outputs)
    #assert False
    return outputs


class MyDataset(Dataset):
    def __init__(self, text_list, labels, tokenizer, max_length):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        label_encoding = \
        self.tokenizer(label, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')[
            'input_ids'][0]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_encoding
        }


if __name__ == '__main__':
    model_name = "./t5_large"  # 选择T5模型的规模
    checkpoint_path = r'output/checkpoint-24000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

    with open("GEM_data_translation_ChEBI_0804/test.source", "r", encoding="utf-8") as f:
        lines = f.readlines()
        texts_test = [x[6:].strip() for x in lines]

    with open("GEM_data_translation_ChEBI_0804/test.target", "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels_test = [x.strip() for x in lines]

    test_dataset = MyDataset(texts_test, labels_test, tokenizer, 512)
    batch_size = 8
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    with open('ChEBI_t5_large.csv', 'a', newline='') as csvfile, \
            open('ChEBI_t5_large_output.csv', 'a', newline='') as csvfile2:
        writer = csv.writer(csvfile)
        writer2 = csv.writer(csvfile2)
        writer.writerow(['gold', 'predict'])
        outputs = []
        for batch_idx, samples in tqdm(enumerate(test_dataloader)):
            # batch是一个字典或元组，包含了批次的数据
            output = generate_text(samples)
            outputs += output
            writer2.writerows([output])
        zipper = list(zip(labels_test, outputs))
        writer.writerows(zipper)





