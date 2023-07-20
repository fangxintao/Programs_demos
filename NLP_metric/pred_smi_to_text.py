from mindformers import T5ForConditionalGeneration, T5Tokenizer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig
from mindformers import build_tokenizer
from mindformers import build_model
from mindformers.tools.register import MindFormerConfig
from mindspore import nn as nn
from mindspore import ops as ops
import mindspore as ms
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
import os.path as osp
# from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

def pred_one(input_str_ls):
    input_ids = []
    for input_str in input_str_ls:
        input_tokens = voc.tokenizer_for_downstream_source(input_str)
        input_id = voc.encode(input_tokens, max_len=512)
        input_ids.append(input_id.tolist())
    output_ls, _ = model.generate(input_ids, do_sample=False)
    outputs = []
    for output in output_ls:
        try:
            output_tokens = voc.decode(output[1:-1])
        except:
            print('error', output[1:-1])
            assert False
        outputs.append(output_tokens.strip())
    return outputs




def pred_one_old(input_str):
    input_tokens = voc.tokenizer_for_downstream_source(input_str)
    input_ids = voc.encode(input_tokens, max_len=512)

    output, log_prod = model.generate(input_ids, do_sample=False)

    output = tokenizer.decode(output, skip_special_tokens=True)

    return output


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
            # line = int(line)
            line = list(map(float, line.strip().split()))
            data.append(line)
    return data


def metrics(infile, outfile):
    in_ls = read_text_source(infile)[:16]
    out_ls = read_text_source(outfile)[:16]
    BATCH_SIZE = 1
    len_sample = len(in_ls)

    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1

    # bleu_scores = []
    meteor_scores = []
    references = []
    hypotheses = []
    outputs = []
    print('长度:', len_sample)
    for i in tqdm(range(len_batch)):
        in_s = in_ls[i * 16: 16 * (i + 1)]
        gt_16 = out_ls[i * 16: 16 * (i + 1)]
        output_16 = pred_one(in_s)
        zipper = list(zip(in_s, gt_16, output_16))
        for smi, gt, output in zipper:
            outputs.append((smi, gt, output))
        print('outputs', outputs[0])
    assert False

    for i, (smi, gt, out) in enumerate(outputs):
        gt_tokens = tokenizer(gt, truncation=True, max_length=512,
                                            padding='max_length', return_tensors='ms')['input_ids']
        gt_tokens = tokenizer.convert_ids_to_tokens(gt_tokens.asnumpy().tolist())
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('</s>').__ne__, gt_tokens))
        # print('gt_tokens', gt_tokens)
        # print(type(gt_tokens))

        out_tokens = tokenizer(out, truncation=True, max_length=512,
                                             padding='max_length', return_tensors='ms')['input_ids']
        out_tokens = tokenizer.convert_ids_to_tokens(out_tokens.asnumpy().tolist())
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('</s>').__ne__, out_tokens))
        # print('out_tokens', out_tokens)
        # print(type(out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)
        # assert False
    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    print('Average Meteor score:', _meteor_score)

    # ------------------------------------------------------------------------------

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = []

    for i, (smi, gt, out) in enumerate(outputs):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)
    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score
    # return _meteor_score

def draw_plot(epoch, name, eval_acc, test_acc):
    print(name, eval_acc, test_acc)
    # plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title(name)
    plt.xlabel("epochs")
    plt.ylabel(name)
    plt.legend(["eval_acc",
                "test_acc"])
    plt.show()


if __name__ == '__main__':
    nltk.download('corpora/wordnet')
    nltk.download('omw-1.4')
    # dataset_name = 'data_t5mol'
    # print('dataset_name', dataset_name)
    p = r'data_downstream/data_pubchem_iupac_smi'
    # tr_infile = f'./{p}/{dataset_name}/train_{dataset_name}_smiles.txt'
    # tr_outfile = f'./{p}/{dataset_name}/train_{dataset_name}_labels.txt'
    eval_infile = f'./{p}/dev_smiles.txt'
    eval_outfile = f'./{p}/dev_iupac.txt'
    test_infile = f'./{p}/test_smiles.txt'
    test_outfile = f'./{p}/test_iupac.txt'

    random.seed(666)
    context_config = ContextConfig(device_id=6, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)

    config = MindFormerConfig('configs/t5/run_t5_small_on_wmt16.yaml')
    model = build_model(config.model)
    # tokenizer = build_tokenizer(config.processor.tokenizer)
    tokenizer = T5Tokenizer.from_pretrained("t5_small")

    checkpoint_path = r'./output/rank_0/checkpoint/'
    # checkpoint_path = r'./output/rank_0/checkpoint/'
    checkpoints = os.listdir(checkpoint_path)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[1] == '.ckpt']
    checkpoints = sorted(checkpoints,
                         key=lambda x: (int(x.split('_')[7].split('-')[1]), int(x.split('_')[8].split('.')[0])))
    model.set_train(False)

    bleu2s = []
    bleu4s = []
    rouge_1s = []
    rouge_2s = []
    rouge_ls = []
    _meteor_scores = []

    bleu2s_t = []
    bleu4s_t = []
    rouge_1s_t = []
    rouge_2s_t = []
    rouge_ls_t = []
    _meteor_scores_t = []

    epochs = []
    for i, checkpoint in enumerate(checkpoints):
        if i % 2 == 0:
            continue
        # if i > 4:
        #     break
        epochs.append(i)

        # tmp = 'fangxt_t5_01sum6shuffle_rank_0-5_6272.ckpt'
        # config.checkpoint_name_or_path = os.path.join(checkpoint_path, tmp)

        config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)
        print('checkpoint_name_or_path', config.checkpoint_name_or_path)
        model.load_checkpoint(config)
        # _meteor_score = metrics(eval_infile, eval_outfile)
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score = metrics(eval_infile, eval_outfile)
        bleu2s.append(bleu2)
        bleu4s.append(bleu4)
        rouge_1s.append(rouge_1)
        rouge_2s.append(rouge_2)
        rouge_ls.append(rouge_l)
        _meteor_scores.append(_meteor_score)

        # _meteor_score_t = metrics(test_infile, test_outfile)
        bleu2_t, bleu4_t, rouge_1_t, rouge_2_t, rouge_l_t, _meteor_score_t = metrics(test_infile, test_outfile)
        bleu2s_t.append(bleu2_t)
        bleu4s_t.append(bleu4_t)
        rouge_1s_t.append(rouge_1_t)
        rouge_2s_t.append(rouge_2_t)
        rouge_ls_t.append(rouge_l_t)
        _meteor_scores_t.append(_meteor_score_t)

    # input_data = list(zip(bleu2s, bleu4s, rouge_1s, rouge_2s, rouge_ls, _meteor_scores,
    #                       bleu2s_t, bleu4s_t, rouge_1s_t, rouge_2s_t, rouge_ls_t, _meteor_scores_t))
    #
    # with open(os.path.join(checkpoint_path, f'pred_{dataset_name}.csv'), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(input_data)

    draw_plot(epochs, 'BLUE_2', bleu2s, bleu2s_t)
    draw_plot(epochs, 'BLUE_4', bleu4s, bleu4s_t)
    draw_plot(epochs, 'ROUGE_1', rouge_1s, rouge_1s_t)
    draw_plot(epochs, 'ROUGE_2', rouge_2s, rouge_2s_t)
    draw_plot(epochs, 'ROUGE_L', rouge_ls, rouge_ls_t)
    draw_plot(epochs, 'METEOR', _meteor_scores, _meteor_scores_t)


