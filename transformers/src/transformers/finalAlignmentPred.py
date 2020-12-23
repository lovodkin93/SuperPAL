import pandas as pd
import os
import sys
import numpy as np
import pickle
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nlp/ernstor1/alignmentEval/')
from annotation_checker_pkg.annotation_checker_task3 import Comparator

def calc_final_alignments(csv_path, model_path, preds, preds_prob):
    OUT_PATH = '/home/nlp/ernstor1/alignmentEval/annotation_checker_pkg/Task3_Datasets/'

    df = pd.read_csv(os.path.join(csv_path,'dev.tsv'), sep='\t')
    positiveAlignments = df#[preds==1]
    positiveAlignments = positiveAlignments[['database', 'topic','summaryFile', 'scuSentCharIdx', 'scuSentence',
                                             'documentFile',	'docSentCharIdx', 'docSentText', 'docSpanOffsets',
                                             'summarySpanOffsets', 'docSpanText', 'summarySpanText','Quality']]
    positiveAlignments['pred_prob'] = preds_prob#[preds==1]
    pred_file_name = csv_path[:-1].split('/')[-1] + '_' + model_path[:-1].split('/')[-1] + '_negative' + '.csv'
    pred_out_path = os.path.join(OUT_PATH, pred_file_name)
    positiveAlignments.to_csv(pred_out_path, index=False)

    gold_path = "/home/nlp/ernstor1/alignmentEval/annotation_checker_pkg/Task3_Datasets/finalAlignmentDataset_dev_cleaned_wo_duplications.csv"


    Comparator().compare_with_hard(gold_path, pred_out_path, hard_threshold=0.25, ignore_missing_from_gold=False)



def calc_alignment_sim_mat(csv_path, model_path, preds_prob):
    OUT_PATH = '/home/nlp/ernstor1/main_summarization/sim_mats/'

    df = pd.read_csv(os.path.join(csv_path,'dev.tsv'), sep='\t')
    spans_num = int(np.sqrt(len(df)))
    sim_mat = np.zeros((spans_num, spans_num))
    sim_mat_idx = df[['sim_mat_idx']]
    for sim_idx, prob in zip(sim_mat_idx.values, preds_prob):
        sim_idx = sim_idx[0].split(',')
        sim_mat[int(sim_idx[0]),int(sim_idx[1])] = prob
    pred_file_name = 'SupAligner' + '_' + model_path[:-1].split('/')[-1] + '_' + df['topic'].iloc[0] + '.pickle'
    pred_out_path = os.path.join(OUT_PATH, pred_file_name)

    with open(pred_out_path, 'wb') as handle:
        pickle.dump(sim_mat, handle)


if __name__ == "__main__":
    calc_alignment_sim_mat('/home/nlp/ernstor1/transformers/data/newMRPC_OIU/devTmp/', '', np.zeros(422))


