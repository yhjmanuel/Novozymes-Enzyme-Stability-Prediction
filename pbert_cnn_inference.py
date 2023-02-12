import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import tqdm
import Levenshtein
from scipy.stats import rankdata
from transformers import BertTokenizer, BertModel
import pickle
from torch.utils.data import Dataset, DataLoader
from pbert_cnn_train import *

def conv_bert_inference(conv_device, bert_device, model_dir,
                        test_csv='test.csv', test_dir='nesp_features.npy'):
    def gen_mutations(name, df,
                      wild="VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQ""RVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGT""NAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKAL""GSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"):
        result = []
        for _, r in df.iterrows():
            ops = Levenshtein.editops(wild, r.protein_sequence)
            assert len(ops) <= 1
            if len(ops) > 0 and ops[0][0] == 'replace':
                idx = ops[0][1]
                result.append([ops[0][0], idx + 1, wild[idx], r.protein_sequence[idx]])
            elif len(ops) == 0:
                result.append(['same', 0, '', ''])
            elif ops[0][0] == 'insert':
                assert False, "Ups"
            elif ops[0][0] == 'delete':
                idx = ops[0][1]
                result.append(['delete', idx + 1, wild[idx], '-'])
            else:
                assert False, "Ups"

        df = pd.concat([df, pd.DataFrame(data=result, columns=['op', 'idx', 'wild', 'mutant'])], axis=1)
        df['mut'] = df[['wild', 'idx', 'mutant']].astype(str).apply(lambda v: ''.join(v), axis=1)
        df['name'] = name
        return df

    MAX_LENGTH = 512
    PRETRAINED_BERT = 'Rostlab/prot_bert'

    model = Model(conv_structure=[14, 16, 32, 64, 80, 96, 128], fc1_structure=[3072, 512, 1],
                  custom_bert=CustomBert(), conv_device=conv_device, bert_device=bert_device)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
    model.load_state_dict(torch.load(model_dir))
    model.device_transfer()
    model.eval()
    df_test = gen_mutations('wildtypeA', pd.read_csv('test.csv'))
    df_test = df_test.loc[df_test.op == 'replace'].reset_index()
    results = []
    test_data = np.load(test_dir)
    #     with torch.no_grad():
    #         for instance in test_data:
    #             instance = torch.tensor(instance, dtype=torch.float32).to(device)
    #             # 0-use dt
    #             # 1-use ddG
    #             result = model(instance)[1].flatten().item()
    #             results.append(result)
    with torch.no_grad():
        for i in range(len(df_test)):
            r = df_test.iloc[i]
            wt = r['protein_sequence'][: int(r.idx)] + r['mutant'] + r['protein_sequence'][int(r.idx) + 1:]
            mut = r['protein_sequence']
            wt_enc = tokenizer(' '.join(wt), padding='max_length', max_length=MAX_LENGTH,
                               return_token_type_ids=False, truncation=True, return_tensors='pt')
            mut_enc = tokenizer(' '.join(mut), padding='max_length', max_length=MAX_LENGTH,
                                return_token_type_ids=False, truncation=True, return_tensors='pt')
            mut_mask = torch.zeros(MAX_LENGTH)
            if r.idx >= MAX_LENGTH:
                mut_mask[MAX_LENGTH] = 1
            else:
                mut_mask[r.idx] = 1
            result = model(torch.tensor(test_data[i], dtype=torch.float32), wt_enc, mut_enc, mut_mask, att_mask='one_hot')[0].flatten().item()
            results.append(result)
    return results


if __name__ == '__main__':
    results = conv_bert_inference(conv_device='cpu', bert_device='mps',
                                  model_dir='cnn_bert_model.pt')
    df = pd.DataFrame()
    df['pred'] = results
    # the predicted DDGs for single-point mutations in the test set, not the final submission
    df.to_csv('submission_to_be_processed.csv', index=False)