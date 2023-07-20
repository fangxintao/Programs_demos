from mindformers import T5ForConditionalGeneration, T5Tokenizer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig
from mindformers import build_tokenizer
from mindformers import build_model
from mindformers.tools.register import MindFormerConfig
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint
import csv
from t5_mask_tokenizer import Vocabulary
import re
import time
import sentencepiece as spm


# def pred_one(input_str_ls):
#     input_ids = []
#     for input_str in input_str_ls:
#         input_tokens = voc.tokenizer_for_downstream_source(input_str)
#         input_id = voc.encode(input_tokens, max_len=512)
#         input_ids.append(input_id.tolist())
#     output_ls, _ = model.generate(input_ids, do_sample=False)
#     outputs = []
#     for output in output_ls:
#         try:
#             output_tokens = voc.decode(output[1:-1])
#         except:
#             print('error', output[1:-1])
#             continue
#         outputs.append(output_tokens.strip())
#     return outputs

def pred_one(input_str_ls):
    # print(input_str_ls)
    input_ids = []
    for input_str in input_str_ls:
        input_id = sp.encode_as_ids(input_str)
        input_id = input_id + [1] + [0] * (511-len(input_id)) if len(input_id) <= 511 else input_id[:511] + [1]
        input_ids.append(input_id)
    output_ls, _ = model.generate(input_ids, do_sample=False)
    outputs = []
    for output in output_ls:
        try:
            print(output)
            # output_tokens = voc.decode(output[1:-1])
            output_tokens = sp.Decode(output[7:-8].tolist())
        except:
            print('error', output)
            continue
        outputs.append(output_tokens.strip())
  
    return outputs


def read_text_source(train_file):
    """Read the text files and return a list."""
    with open(train_file) as fp:
        data = []
        for line in fp:
            line = line.strip()
            if line:
                data.append(line)
    return data


def print_acc(infile, outfile):
    in_ls = read_text_source(infile)[:100]
    out_ls = read_text_source(outfile)[:100]
    BATCH_SIZE = 16
    len_sample = len(in_ls)

    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1

    # random = [randint(0, len_source) for _ in range(100)]
    # for i in random:
    #     in_ls.append(in_lines[i].rstrip())
    #     out_ls.append(out_lines[i].rstrip())
    len_sample = len(in_ls)
    print('长度:', len_sample)
    acc = 0
    with open(os.path.join(checkpoint_path, f'pred_accuracy_gold_predict_test_sci.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name] )
        writer.writerow(['gold', 'predict'])
        for i in tqdm(range(len_batch)):
            in_s = in_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            out_s = out_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            out = pred_one(in_s)
            zipper = list(zip(out, out_s))
            print(zipper)
            assert False
            writer.writerows(zipper)
            for predict, gold in zipper:
                gold1 = pattern.findall(gold)[-1]
                # print('预测', predict)
                # print('标签', gold1)
                if predict == gold1:
                    acc += 1      
    acc_avg = acc / len_sample
    return acc_avg


def draw_plot(epoch, train_acc, eval_acc, test_acc):
    # plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plt.legend(["train_acc",
    #             "eval_acc",
    #             "test_acc"])
    plt.legend(["eval_acc",
                "test_acc"])
    plt.show()


if __name__ == '__main__':
    context_config = ContextConfig(device_id=2, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)
    start_time = time.perf_counter()
    list_dataset = ['USPTO']
    pattern = re.compile('▁<SMI_S>▁(.*)▁<SMI_E>▁', re.I)
    sp = spm.SentencePieceProcessor(model_file='./spiece.model')
    for dataset_name in list_dataset:
        print('dataset_name', dataset_name)
        p = r'downstream_eval_datasets/GEM_data_each_simple/Translation'
        eval_infile = f'./{p}/{dataset_name}/dev_{dataset_name}_smiles.txt'
        eval_outfile = f'./{p}/{dataset_name}/dev_{dataset_name}_labels.txt'
        test_infile = f'./{p}/{dataset_name}/test_{dataset_name}_smiles.txt'
        test_outfile = f'./{p}/{dataset_name}/test_{dataset_name}_labels.txt'

        random.seed(666)

        config = MindFormerConfig('configs/t5/run_test_base.yaml')
        model = build_model(config.model)
        voc = Vocabulary("general_plus_element.vocab", "union.model", 'element.txt')

        checkpoint_path = r'/home/ma-user/work/fangxt/checkpoint_0719/checkpoint_scifive_base'
        checkpoints = os.listdir(checkpoint_path)
        checkpoints = [i for i in checkpoints if os.path.splitext(i)[1] == '.ckpt']
        checkpoints = sorted(checkpoints,
                             key=lambda x: (int(x.split('_')[2].split('-')[1]), int(x.split('_')[3].split('.')[0])))
        model.set_train(False)

        train_acc = []
        eval_acc = []
        test_acc = []
        epochs = []
        for i, checkpoint in enumerate(checkpoints):
            # if i > 10 and i % 3 != 0:
            #     continue
            if i % 2 == 1:
                continue
            epoch_ = int(checkpoint.split('_')[2].split('-')[1])
            step = int(checkpoint.split('_')[3].split('.')[0])
            x = epoch_ * 550 + step
            epochs.append(x)
            config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)
            # tmp = 'fangxt_t5_iupac_to_smi_10w_rank_0-20_6250.ckpt'
            # config.checkpoint_name_or_path = os.path.join(checkpoint_path, tmp)

            model.load_checkpoint(config)

            # e_acc = print_acc(eval_infile, eval_outfile)
            # # e_acc = print_acc(eval_outfile, eval_infile)
            # print('eval_acc', e_acc)
            # eval_acc.append(e_acc)

            t_acc = print_acc(test_infile, test_outfile)
            # t_acc = print_acc(test_outfile, test_infile)
            print('test_acc', t_acc)
            test_acc.append(t_acc)
            # if i == 0:
            end_time1 = time.perf_counter()
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))

        # print(dataset_name, eval_acc)
        print(dataset_name, test_acc)
        input_data = list(zip(epochs, test_acc))
        print(input_data)
        with open(os.path.join(checkpoint_path, f'pred_accuracy_test.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset_name] )
            writer.writerows(input_data)
    end_time = time.perf_counter()
    print('总计耗时%.2f秒' % (end_time - start_time))
    # draw_plot(epochs, 'F1-score', eval_acc, test_acc)

