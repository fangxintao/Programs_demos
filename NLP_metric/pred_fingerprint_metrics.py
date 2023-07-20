from mindformers import T5ForConditionalGeneration, T5Tokenizer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig
from mindformers import build_tokenizer
from mindformers import build_model
from mindformers.tools.register import MindFormerConfig
from mindspore import nn as nn
from mindspore import ops as ops
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import mindspore as ms
import argparse
import csv
import os.path as osp
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import csv
from t5_mask_tokenizer import Vocabulary
import re
import time



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
    in_ls = read_text_source(infile)[2000:2100]
    out_ls = read_text_source(outfile)[2000:2100]
    BATCH_SIZE = 16
    len_sample = len(in_ls)

    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1

    bad_mols = 0
    outputs = []
    print('长度:', len_sample)
    with open(os.path.join(checkpoint_path, f'pred_ChEBI_finger_gold_predict_test.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name])
        writer.writerow(['gold', 'predict'])
        for i in tqdm(range(len_batch)):
            in_s = in_ls[i * BATCH_SIZE: BATCH_SIZE * (i + 1)]
            gt_16 = out_ls[i * BATCH_SIZE: BATCH_SIZE * (i + 1)]
            output_16 = pred_one(in_s)
            zipper = list(zip(in_s, gt_16, output_16))
            zipper_tmp = list(zip(gt_16, output_16))
            writer.writerows(zipper_tmp)
            for des, gt, output in zipper:
                gt = pattern.findall(gt)[-1]
                # print('gold', gt)
                # print('output', output)
                try:
                    gt_smi = gt
                    ot_smi = output
                    gt_m = Chem.MolFromSmiles(gt_smi)
                    ot_m = Chem.MolFromSmiles(ot_smi)
                    if ot_m == None: raise ValueError('Bad SMILES')
                    outputs.append((des, gt_m, ot_m))
                except:
                    bad_mols += 1
               
    validity_score = len(outputs) / (len(outputs) + bad_mols)
    print('validity:', validity_score)
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    print('Average MACCS Similarity:', maccs_sims_score)
    print('Average RDK Similarity:', rdk_sims_score)
    print('Average Morgan Similarity:', morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score


def draw_plot(epoch, name, eval_acc, test_acc):
    print(name, eval_acc, test_acc)
    # plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title(name)
    plt.xlabel("epochs")
    plt.legend(["eval_acc",
                "test_acc"])
    plt.show()


if __name__ == '__main__':
    start_time = time.perf_counter()
    dataset_name = 'ChEBI'
    print('dataset_name', dataset_name)
    morgan_r = 2
    # p = r'downstream_eval_datasets/data_each_simple'
    p = r'downstream_eval_datasets/GEM_data_each_simple/Translation'
    pattern = re.compile('▁<SMI_S>▁(.*)▁<SMI_E>▁', re.I)
    # tr_infile = f'./{p}/{dataset_name}/train_{dataset_name}_smiles.txt'
    # tr_outfile = f'./{p}/{dataset_name}/train_{dataset_name}_labels.txt'
    eval_infile = f'./{p}/{dataset_name}/dev_{dataset_name}_smiles.txt'
    eval_outfile = f'./{p}/{dataset_name}/dev_{dataset_name}_labels.txt'
    test_infile = f'./{p}/{dataset_name}/test_{dataset_name}_smiles.txt'
    test_outfile = f'./{p}/{dataset_name}/test_{dataset_name}_labels.txt'

    random.seed(666)
    context_config = ContextConfig(device_id=5, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)

    config = MindFormerConfig('configs/t5/run_test_base.yaml')
    model = build_model(config.model)
    voc = Vocabulary("general_plus_element.vocab", "union.model", 'element.txt')

    checkpoint_path = r"/home/ma-user/work/fangxt/checkpoint_0716/checkpoint_base"
    checkpoints = os.listdir(checkpoint_path)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[1] == '.ckpt']
    checkpoints = sorted(checkpoints,
                         key=lambda x: (int(x.split('_')[2].split('-')[1]), int(x.split('_')[3].split('.')[0])))
    model.set_train(False)
    validity_score_dev = []
    MACCS_sims_dev = []
    RDK_sims_dev = []
    morgan_sims_dev = []

    validity_score_test = []
    MACCS_sims_test = []
    RDK_sims_test = []
    morgan_sims_test = []

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

        # tmp = 'fangxt_t5_01sum6shuffle_rank_0-5_6272.ckpt'
        # config.checkpoint_name_or_path = os.path.join(checkpoint_path, tmp)

        config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)
        print('checkpoint_name_or_path', config.checkpoint_name_or_path)
        model.load_checkpoint(config)

        # validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = metrics(eval_infile, eval_outfile)
        # validity_score_dev.append(validity_score)
        # MACCS_sims_dev.append(maccs_sims_score)
        # RDK_sims_dev.append(rdk_sims_score)
        # morgan_sims_dev.append(morgan_sims_score)


        validity_score_t, maccs_sims_score_t, rdk_sims_score_t, morgan_sims_score_t = metrics(test_infile, test_outfile)
        print('test result')
        validity_score_test.append(validity_score_t)
        MACCS_sims_test.append(maccs_sims_score_t)
        RDK_sims_test.append(rdk_sims_score_t)
        morgan_sims_test.append(morgan_sims_score_t)
        
        end_time1 = time.perf_counter()
        print('单个耗时%.2f秒' % (end_time1 - start_time))
        print('单个耗时%.2f秒' % (end_time1 - start_time))
        print('单个耗时%.2f秒' % (end_time1 - start_time))

    # input_data = list(zip(epochs, validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score,
    #                      validity_score_t, maccs_sims_score_t, rdk_sims_score_t, morgan_sims_score_t))
    input_data = list(zip(epochs, validity_score_test, MACCS_sims_test, RDK_sims_test, morgan_sims_test))
    
    with open(os.path.join(checkpoint_path, f'pred_ChEBI_finger_test.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name])
        writer.writerows(input_data)
    end_time = time.perf_counter()
    print('总计耗时%.2f秒' % (end_time - start_time))

    # draw_plot(epochs, 'validity_score', validity_score_dev, validity_score_test)
    # draw_plot(epochs, 'MACCS_sims', MACCS_sims_dev, MACCS_sims_test)
    # draw_plot(epochs, 'RDK_smis', RDK_sims_dev, RDK_sims_test)
    # draw_plot(epochs, 'MORGAN_sims', morgan_sims_dev, morgan_sims_test)




