# from mindformers import T5ForConditionalGeneration, T5Tokenizer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig
from mindformers import build_tokenizer
from mindformers import build_model
from mindformers.tools.register import MindFormerConfig
import mindspore.common.dtype as mstype
import mindspore as ms
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import csv
from t5_mask_tokenizer import Vocabulary
import time
import sentencepiece as spm

voc = Vocabulary("general_plus_element.vocab", "union.model", 'element.txt')
# def eval_batch(input_str_ls, label_str_ls):
#     assert len(input_str_ls) == len(label_str_ls)
#     zipper = zip(input_str_ls, label_str_ls)
#     input_ids = []
#     attention_mask = []
#     model_input = []
#     for input_str, label_str in zipper:
#         input_tokens = voc.tokenizer_for_downstream_source(input_str)
#         input_id = voc.encode(input_tokens, max_len=512)
#         input_ids.append(input_id)
#         attention_mask_one = [0 if id == 0 else 1 for id in input_id.tolist()]
#         attention_mask.append(attention_mask_one)
#         label_token = voc.tokenizer_for_downstream_target(label_str)
#         model_input_one = voc.encode(label_token, max_len=256)
#         model_input.append(model_input_one)

#     input_ids = ms.Tensor(input_ids, mstype.int32)
#     attention_mask = ms.Tensor(attention_mask, mstype.int32)
#     model_input = ms.Tensor(model_input, mstype.int32)
#     output = model(input_ids, attention_mask, model_input, return_loss=True)
#     return output

def eval_batch(input_str_ls, label_str_ls):
    assert len(input_str_ls) == len(label_str_ls)
    zipper = zip(input_str_ls, label_str_ls)
    input_ids = []
    attention_mask = []
    model_input = []
    for input_str, label_str in zipper:
        # input_tokens = voc.tokenizer_for_downstream_source(input_str)
        # input_id = voc.encode(input_tokens, max_len=512)
        input_id = sp.encode_as_ids(input_str)
        input_id = input_id + [1] + [0] * (511-len(input_id)) if len(input_id) <= 511 else input_id[:511] + [1]
        input_ids.append(input_id)
        attention_mask_one = [0 if id == 0 else 1 for id in input_id]
        attention_mask.append(attention_mask_one)
        # label_token = voc.tokenizer_for_downstream_target(label_str)
        # model_input_one = voc.encode(label_token, max_len=256)
        
        model_input_one = sp.encode_as_ids(label_str)
        model_input_one = model_input_one + [1] + [0] * (255-len(model_input_one)) if len(model_input_one) <= 255 else model_input_one[:255] + [1]
        
        model_input.append(model_input_one)

    input_ids = ms.Tensor(input_ids, mstype.int32)
    attention_mask = ms.Tensor(attention_mask, mstype.int32)
    model_input = ms.Tensor(model_input, mstype.int32)
    output = model(input_ids, attention_mask, model_input, return_loss=True)
    return output


def read_text_source(train_file):
    """Read the text files and return a list."""
    with open(train_file, 'r', encoding='utf-8') as fp:
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


def print_loss(infile, outfile):
    in_ls = read_text_source(infile)
    out_ls = read_text_source(outfile)
    zipper = list(zip(in_ls, out_ls))
    zipper_2000 = random.sample(zipper, 2000)
    in_ls, out_ls = zip(*zipper_2000)
    # print(len(in_ls))
    # print(len(out_ls))
    # print(in_ls[:1])
    # print(out_ls[:1])
    # assert False
    BATCH_SIZE = 16
    len_sample = len(in_ls)
    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1

    totol_loss = 0
    for i in range(len_batch):
        in_s = in_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
        out_s = out_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
        batch_loss = eval_batch(in_s, out_s)
        totol_loss += batch_loss
    loss_avg = float(totol_loss / len_batch)
    return loss_avg


def print_loss2(infile, outfile):
    pass


def draw_plot(epoch, train_acc, eval_acc, test_acc):
    plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title("Loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend(["train_loss",
                "eval_loss",
                "test_loss"])
    print('show the plot')
    plt.show()


if __name__ == '__main__':
    start_time = time.process_time()
    p = r'downstream_eval_datasets/GEM_data_all_simple'
    tr_infile = f'./{p}/train.source'
    tr_outfile = f'./{p}/train.target'
    eval_infile = f'./{p}/dev.source'
    eval_outfile = f'./{p}/dev.target'
    test_infile = f'./{p}/test.source'
    test_outfile = f'./{p}/test.target'

    random.seed(666)
    context_config = ContextConfig(device_id=2, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)

    config = MindFormerConfig('configs/t5/run_test_base.yaml')
    model = build_model(config.model)
    voc = Vocabulary("general_plus_element.vocab", "union.model", 'element.txt')
    sp = spm.SentencePieceProcessor(model_file='./spiece.model')
    # checkpoint_path = r'/data/fangxt/transformer2/evaluation_downstream/checkpoint/checkpoint_0703/checkpoint_simple_base'
    checkpoint_path = r"/home/ma-user/work/fangxt/checkpoint_0719/checkpoint_scifive_base"
    checkpoints = os.listdir(checkpoint_path)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[1] == '.ckpt']

    checkpoints = sorted(checkpoints,
                         key=lambda x: (int(x.split('_')[2].split('-')[1]), int(x.split('_')[3].split('.')[0])))

    train_loss = []
    eval_loss = []
    test_loss = []
    epochs = []

    model.set_train(False)
    for i, checkpoint in enumerate(checkpoints):
        # if i % 2 == 1:
        #     continue
        epoch_ = int(checkpoint.split('_')[2].split('-')[1])
        step = int(checkpoint.split('_')[3].split('.')[0])
        x = epoch_ * 550 + step
        epochs.append(x)
        config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)
        print('checkpoint_name_or_path', config.checkpoint_name_or_path)
        # model = T5ForConditionalGeneration.from_pretrained_xhliu(
        #     pretrained_model_name_or_dir=config.checkpoint_name_or_path,
        #     config_yaml='/data/fangxt/transformer2_fangxt/configs/t5/model_config/t5_small.yaml')
        model.load_checkpoint(config)
        tr_loss = print_loss(tr_infile, tr_outfile)
        # tr_loss = print_loss(tr_outfile, tr_infile)
        print('train_loss:', tr_loss)
        train_loss.append(tr_loss)

        e_loss = print_loss(eval_infile, eval_outfile)
        # e_loss = print_loss(eval_outfile, eval_infile)
        print('eval_loss', e_loss)
        eval_loss.append(e_loss)

        t_loss = print_loss(test_infile, test_outfile)
        # t_loss = print_loss(test_outfile, test_infile)
        print('test_loss', t_loss)
        test_loss.append(t_loss)
        end_time1 = time.perf_counter()
        print('单个耗时%.2f秒' % (end_time1 - start_time))
        print('单个耗时%.2f秒' % (end_time1 - start_time))
        print('单个耗时%.2f秒' % (end_time1 - start_time))

    print(train_loss)
    print(eval_loss)
    print(test_loss)
    input_data = list(zip(epochs, train_loss, eval_loss, test_loss))
    with open(os.path.join(checkpoint_path, 'loss.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['steps', 'eval_loss', 'test_loss'])
        writer.writerows(input_data)

    draw_plot(epochs, train_loss, eval_loss, test_loss)
    end_time = time.process_time()
    print('总计耗时%.2f秒' % (end_time - start_time))


