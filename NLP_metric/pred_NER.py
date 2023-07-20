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
from sklearn.metrics import f1_score
import re
import time
from t5_mask_tokenizer import Vocabulary
import csv

def pred_one(input_str_ls):
    # print(input_str_ls)
    input_ids = []
    for input_str in input_str_ls:
        input_tokens = voc.tokenizer_for_downstream_source(input_str)
        input_id = voc.encode(input_tokens, max_len=512)
        input_ids.append(input_id.tolist())
    # s_time = time.process_time()
    output_ls, _ = model.generate(input_ids, do_sample=False)
    # e_time = time.process_time()
    # print(output_ls)
    # print(len(output_ls[-1]))
    # print('总计耗时%.2fhao秒' % ((e_time - s_time) * 1000))
    outputs = []
    for output in output_ls:
        try:
            output_tokens = voc.decode(output[1:-1])
        except:
            print('error', output[1:-1])
            continue
        outputs.append(output_tokens.strip())
    # e_time2 = time.process_time()
    # print(outputs)
    # print('endtime2总计耗时%.2fhao秒' % ((e_time2 - s_time)*1000))
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


def My_f1_score(A, B, C):
    f1, p, r = 2 * A / (B + C), A / B, A / C
    return f1


def print_acc(infile, outfile):
    in_ls = read_text_source(infile)[:1000]
    out_ls = read_text_source(outfile)[:1000]
    # in_ls = [r'BC2GM:RESULTS : Factors associated with significantly ( P < . 05 ) increased risk of treatment failure in a Cox multivariate analysis included age older than 45 years ( relative hazard , 1 . 17 ; 95 % confidence interval [ CI ] , 1 . 02 - 1 . 33 ) , Karnofsky performance score less than 90 % ( 1 . 27 ; 95 % CI , 1 . 07 - 1 . 51 ) , absence of hormone receptors ( 1 . 31 ; 95 % CI , 1 . 15 - 1 . 51 ) , prior use of adjuvant chemotherapy ( 1 . 31 ; 95 % CI , 1 . 10 - 1 . 56 ) , initial disease - free survival interval after adjuvant treatment of no more than 18 months ( 1 . 99 ; 95 % CI , 1 . 62 - 2 . 43 ) , metastases in the liver ( 1 . 47 ; 95 % CI , 1 . 20 - 1 . 80 ) or central nervous system ( 1 . 56 ; 95 % CI , 0 . 99 - 2 . 46 [ approaches significance ] ) vs soft tissue , bone , or lung , 3 or more sites of metastatic disease ( 1 . 32 ; 95 % CI , 1 . 13 - 1 . 54 ) , and incomplete response vs complete response to standard - dose chemotherapy ( 1 . 65 ; 95 % CI , 1 . 36 - 1 . 99 ) .']
    # out_ls = [r'RESULTS : Factors associated with significantly ( P < . 05 ) increased risk of treatment failure in a Cox multivariate analysis included age older than 45 years ( relative hazard , 1 . 17 ; 95 % confidence interval [ CI ] , 1 . 02 - 1 . 33 ) , Karnofsky performance score less than 90 % ( 1 . 27 ; 95 % CI , 1 . 07 - 1 . 51 ) , absence of hormone receptors ( 1 . 31 ; 95 % CI , 1 . 15 - 1 . 51 ) , prior use of adjuvant chemotherapy ( 1 . 31 ; 95 % CI , 1 . 10 - 1 . 56 ) , initial disease - free survival interval after adjuvant treatment of no more than 18 months ( 1 . 99 ; 95 % CI , 1 . 62 - 2 . 43 ) , metastases in the liver ( 1 . 47 ; 95 % CI , 1 . 20 - 1 . 80 ) or central nervous system ( 1 . 56 ; 95 % CI , 0 . 99 - 2 . 46 [ approaches significance ] ) vs soft tissue , bone , or lung , 3 or more sites of metastatic disease ( 1 . 32 ; 95 % CI , 1 . 13 - 1 . 54 ) , and incomplete response vs complete response to standard - dose chemotherapy ( 1 . 65 ; 95 % CI , 1 . 36 - 1 . 99 ) .']

    BATCH_SIZE = 8
    len_sample = len(in_ls)

    len_batch = len_sample % BATCH_SIZE
    if len_batch == 0:
        len_batch = len_sample // BATCH_SIZE
    else:
        len_batch = len_sample // BATCH_SIZE + 1
    print('长度:', len_sample)
    # random = [randint(0, len_source) for _ in range(100)]
    # for i in random:
    #     in_ls.append(in_ls[i].rstrip())
    #     out_ls.append(out_ls[i].rstrip())

    y_true = []
    y_predict = []
    A = 0
    B = 0
    C = 0
    
    with open(os.path.join(checkpoint_path, f'pred_NER_gold_predict_BC4CHEMD.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name] )
        writer.writerow(['gold', 'predict'])
        for i in tqdm(range(len_batch)):
            in_s = in_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            out_s = out_ls[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            out = pred_one(in_s)
            zipper = list(zip(out, out_s))
            writer.writerows(zipper)
            for gold, prediction in zipper:
                res1 = pattern.findall(gold)
                res2 = pattern.findall(prediction)
                tmp_A = len(set(res1) & set(res2))
                tmp_B = len(set(res1))
                tmp_C = len(set(res2))
                A += tmp_A
                B += tmp_B
                C += tmp_C
    f1 = My_f1_score(A, B, C)
    #     for predict, gold in zipper:
    #         y_true.append(gold)
    #         y_predict.append(predict)
    # f1 = f1_score(y_true, y_predit, average='micro')
    return f1


def draw_plot(epoch, name, eval_acc, test_acc):
    # plt.plot(epoch, train_acc, 'r', lw=3)  # lw为曲线宽度
    plt.plot(epoch, eval_acc, 'b', lw=3)
    plt.plot(epoch, test_acc, 'g', lw=3)
    plt.title(name)
    plt.xlabel("epochs")
    plt.ylabel(name)
    # plt.legend(["train_acc",
    #             "eval_acc",
    #             "test_acc"])
    plt.legend(["eval_acc",
                "test_acc"])
    plt.show()


if __name__ == '__main__':
    context_config = ContextConfig(device_id=4, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
    init_context(use_parallel=False, context_config=context_config)
    start_time = time.perf_counter()
    # list_dataset = ['BC2GM', 'BC4CHEMD', 'BC5CDR-chem', 'BC5CDR-disease', 'JNLPBA', 'NCBI-disease']
    list_dataset = ['BC4CHEMD']
    pattern = re.compile('entity\*\s*(.*?)\s*\*entity', re.I)

    for dataset_name in list_dataset:
        # dataset_name = 'esol'
        print('dataset_name', dataset_name)
        p = r'downstream_eval_datasets/GEM_data_each_simple/NER'
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

        checkpoint_path = r'/home/ma-user/work/fangxt/checkpoint_0716/checkpoint_base'
        checkpoints = os.listdir(checkpoint_path)
        checkpoints = [i for i in checkpoints if os.path.splitext(i)[1] == '.ckpt']
        checkpoints = sorted(checkpoints,
                             key=lambda x: (int(x.split('_')[2].split('-')[1]), int(x.split('_')[3].split('.')[0])))
        model.set_train(False)

        train_acc = []
        eval_acc = []
        test_acc = []
        epochs = []
        for i, checkpoint in enumerate(checkpoints[9:10]):
            # if i > 10 and i % 3 != 0:
            #     continue
            epoch_ = int(checkpoint.split('_')[2].split('-')[1])
            step = int(checkpoint.split('_')[3].split('.')[0])
            x = epoch_ * 550 + step
            epochs.append(x)

            # tmp = 'fangxt_t5_iupac_to_smi_10w_rank_0-20_6250.ckpt'
            # config.checkpoint_name_or_path = os.path.join(checkpoint_path, tmp)
            config.checkpoint_name_or_path = os.path.join(checkpoint_path, checkpoint)
            print('checkpoint_name_or_path', config.checkpoint_name_or_path)
            model.load_checkpoint(config)

            # e_acc = print_acc(eval_infile, eval_outfile)
            # # e_acc = print_acc(eval_outfile, eval_infile)
            # print('eval_acc', e_acc)
            # eval_acc.append(e_acc)

            t_acc = print_acc(test_infile, test_outfile)
            # t_acc = print_acc(test_outfile, test_infile)
            print('test_acc', t_acc)
            test_acc.append(t_acc)
            
            end_time1 = time.perf_counter()
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))
            print('单个耗时%.2f秒' % (end_time1 - start_time))

        # print(dataset_name, eval_acc)
        print(dataset_name, test_acc)
        input_data = list(zip(epochs, test_acc))
        with open(os.path.join(checkpoint_path, f'pred_NER.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset_name] )
            writer.writerows(input_data)
    end_time = time.perf_counter()
    print('总计耗时%.2f秒' % (end_time - start_time))
        # draw_plot(epochs, 'F1-score', eval_acc, test_acc)


