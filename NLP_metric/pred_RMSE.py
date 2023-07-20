from mindformers import T5ForConditionalGeneration, T5Tokenizer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig
from mindformers import build_tokenizer
from mindformers import build_model
from mindformers.tools.register import MindFormerConfig
from mindspore import nn as nn
from mindspore import ops as ops
import mindspore as ms
import mindspore.common.dtype as mstype
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint
import json
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import mindspore.numpy as npy
from sklearn.metrics import mean_squared_error
import math
import csv
from t5_mask_tokenizer import Vocabulary
import time



def pred_one(input_str_ls):
    # print(input_str_ls)
    input_ids = []
    for input_str in input_str_ls:
        input_tokens = voc.tokenizer_for_downstream_source(input_str)
        input_id = voc.encode(input_tokens, max_len=512)
        input_ids.append(input_id.tolist())
    output_ls, _ = model.generate(input_ids, do_sample=False)
    # print(output_ls)
    outputs = []
    for output in output_ls:
        print(output)
        try:
            output_tokens = voc.decode(output[1:-1])
        except:
            print('error', output[1:-1])
            continue
            # assert False
        print(output[1:-1])
        outputs.append(output_tokens.strip())
    assert False
    # print(outputs)
     # assert False
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


def read_text_target(target_file):
    """Read the text files and return a list."""
    with open(target_file) as fp:
        data = []
        for line in fp:
            if not line:
                raise Exception('数据当前行为空')
            line = list(map(float, line.strip().split()))
            data.append(line)
    return data


def print_acc(infile, outfile):
    in_ls = read_text_source(infile)
    out_ls = read_text_source(outfile)
    BATCH_SIZE = 16
    len_sample = len(in_ls)

    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1
    print('长度:', len_sample)
    acc = 0
    out_label = None
    scores = None
    result = []
    # for i in tqdm(range(len_sample - 1)):
    for i in tqdm(range(len_batch), disable=True):
        in_s = in_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
        out_s = out_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]

        out_s = ms.Tensor(list(map(float, out_s)), dtype=ms.float32)
        # print('out_s', out_s)
        output = pred_one(in_s)
        # print('output', output)
        try:
            pred_score = ms.Tensor(list(map(float, output)), dtype=ms.float32)
            # print('okokokookokokokokokokokokkokokkokok')
        except:
            print('跳过')
            print('*********************************')
            continue
        if scores is None:
            scores = pred_score
            out_label = out_s
        else:
            scores = npy.concatenate((scores, pred_score), axis=0)
            out_label = npy.concatenate((out_label, out_s), axis=0)

    # 采用RMSE评价
    try:
        scores_new = scores.asnumpy()
        out_label_new = out_label.asnumpy()
        MSE = mean_squared_error(out_label_new, scores_new)
        RMSE = math.sqrt(MSE)
        result.append(RMSE)
    except Exception as e:
        print(e)
    if result:
        return result
    else:
        return [0]


def draw_plot(epoch, eval_acc, test_acc):
    # plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title("RMSE")
    plt.xlabel("epochs")
    plt.ylabel("RMSE_score")
    plt.legend(["eval_acc",
                "test_acc"])
    plt.show()


if __name__ == '__main__':
    context_config = ContextConfig(device_id=5, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)
    start_time = time.perf_counter()
    list_dataset = ['esol', 'freesolv', 'lipophilicity']
    # list_dataset = ['lipophilicity']
    for dataset_name in list_dataset:
    # dataset_name = 'esol'
        print('dataset_name', dataset_name)
        # p = r'downstream_eval_datasets/GEM_data_each_simple/Regression'
        p = r'downstream_eval_datasets/GEM_data_each_simple/Regression'
        # tr_infile = f'./{p}/{dataset_name}/train_{dataset_name}_smiles.txt'
        # tr_outfile = f'./{p}/{dataset_name}/train_{dataset_name}_labels.txt'
        eval_infile = f'./{p}/{dataset_name}/dev_{dataset_name}_smiles.txt'
        eval_outfile = f'./{p}/{dataset_name}/dev_{dataset_name}_labels.txt'
        test_infile = f'./{p}/{dataset_name}/test_{dataset_name}_smiles.txt'
        test_outfile = f'./{p}/{dataset_name}/test_{dataset_name}_labels.txt'

        random.seed(666)

        config = MindFormerConfig('configs/t5/run_test_base.yaml')
        model = build_model(config.model)
        voc = Vocabulary("general_plus_element.vocab", "union.model", 'element.txt')
        # tokenizer = build_tokenizer(config.processor.tokenizer)
        # tokenizer = T5Tokenizer.from_pretrained("t5_small")

        # checkpoint_path = r'/home/ma-user/work/fangxt/checkpoint_0705/checkpoint_base_simple/checkpoint_simple_base'
        checkpoint_path = r"/home/ma-user/work/fangxt/checkpoint_0716/checkpoint_base"
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

            epoch_ = int(checkpoint.split('_')[2].split('-')[1])
            step = int(checkpoint.split('_')[3].split('.')[0])
            x = epoch_ * 548 + step
            epochs.append(x)
            # tmp = 'mindformers_rank_0-5_66.ckpt'
            # config.checkpoint_name_or_path = os.path.join(checkpoint_path, tmp)

            config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)

            print('checkpoint_name_or_path', config.checkpoint_name_or_path)
            model.load_checkpoint(config)
            # tr_acc = print_acc(tr_infile, tr_outfile)
            # print('train_acc:', tr_acc)
            # train_acc.append(tr_acc)
            e_acc = print_acc(eval_infile, eval_outfile)[0]
            print('eval_acc', e_acc)
            eval_acc.append(e_acc)
            t_acc = print_acc(test_infile, test_outfile)[0]
            print('test_acc', t_acc)
            test_acc.append(t_acc)
            # if i == 0:
            end_time1 = time.perf_counter()
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))


        # print(train_acc)
        print(dataset_name, eval_acc)
        print(dataset_name, test_acc)
        input_data = list(zip(epochs, eval_acc, test_acc))
        print(input_data)
        with open(os.path.join(checkpoint_path, f'pred_regression.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset_name])
            writer.writerows(input_data)
    
        # draw_plot(epochs, eval_acc, test_acc)
    end_time = time.perf_counter()
    print('总计耗时%.2f秒' % (end_time - start_time))
# 907

