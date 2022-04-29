import pandas as pd
import os
import sys
import numpy as np
import pickle
import json
from operator import itemgetter
from functools import reduce


def check_index_in_spans(spans, ind):
    for span in spans:
        if (ind>=span[0] and ind<=span[1]):
            return True
    return False

def merge_overlaps(span_list):
    reducer = (
        lambda acc, el: acc[:-1:] + [(min(*acc[-1], *el), max(*acc[-1], *el))]
        if acc[-1][1] > el[0]
        else acc + [el]
    )
    reduced_span_list = reduce(reducer, span_list[1::], [span_list[0]])
    return reduced_span_list

def get_global_spans(doc_table_df, isDoc=False, merge=True):
    span_list = []
    spans = set(doc_table_df.docSpanOffsets) if isDoc else set(doc_table_df.summarySpanOffsets)
    for span in spans:
        sub_spans = span.split(';')
        for sub_span in sub_spans:
            lims = [int(elem) for elem in sub_span.split(',')]
            span_list.append(lims)
    span_list = sorted(span_list, key=itemgetter(0))
    if merge:
        merged_span_list = merge_overlaps(span_list)
        merged_span_list = sorted(merged_span_list, key=itemgetter(0))
        return merged_span_list
    else:
        return span_list

def merge_indice(indice):
    indice = sorted(indice)
    merged_spans = []
    start_index = -1
    for i,elem in enumerate(indice):
        if start_index == -1:
            start_index = elem
        elif elem > indice[i-1] + 1:
            merged_spans.append([start_index, indice[i-1]])
            start_index = indice[i]
    merged_spans.append([start_index, indice[-1]])
    return merged_spans

def get_overlapping_indice(curr_span_indice, missing_spans_indice):
    # curr_span_indice = [ind for ind in list(range(curr_span[0], curr_span[1]))]
    overlapping_indice = [ind for ind in curr_span_indice if ind in missing_spans_indice]
    return overlapping_indice


def str_to_indice(span_str):
    indice_list = list()
    sub_spans = span_str.split(';')
    for sub_span in sub_spans:
        lims = [int(elem) for elem in sub_span.split(',')]
        indice_list+=list(range(lims[0], lims[1]+1))
    return indice_list

def get_filling_spans(missing_spans, full_df_spans):
    full_df_spans_indice = [str_to_indice(span_str) for span_str in full_df_spans]
    missing_spans_indice = [ind for elem in missing_spans for ind in list(range(elem[0], elem[1]+1))]

    # all_filling_spans = [span for span in full_df_spans if any(check_index_in_spans(missing_spans, ind) for ind in list(range(span[0], span[1])))]
    chosen_filling_spans = list()
    while missing_spans_indice:
        overlap_len_list = [len(get_overlapping_indice(curr_indice_list, missing_spans_indice)) for curr_indice_list in full_df_spans_indice]
        if not any(elem>0 for elem in overlap_len_list): # no overlapping
            break
        max_overlap_span_ind = overlap_len_list.index(max(overlap_len_list))
        chosen_filling_spans.append(full_df_spans[max_overlap_span_ind])
        missing_spans_indice = [ind for ind in missing_spans_indice if not ind in full_df_spans_indice[max_overlap_span_ind]]
    return chosen_filling_spans


def get_filling_df_rows(summaries_len_dict, positiveAlignments, df, preds_prob):
    df_with_preds = df.copy()
    df_with_preds['pred_prob'] = preds_prob
    for topic in summaries_len_dict['summaries lens'].keys():
        sub_df = positiveAlignments.loc[positiveAlignments['topic'] == topic]
        covered_spans = get_global_spans(sub_df, False)
        missing_indice = [ind for ind in list(range(summaries_len_dict['summaries lens'][topic])) if
                         not check_index_in_spans(covered_spans, ind)]
        missing_spans = merge_indice(missing_indice)




        full_df_spans = list(set(df.loc[df['topic'] == topic].summarySpanOffsets))
        all_filling_spans = get_filling_spans(missing_spans, full_df_spans)

        extra_rows = list()
        for filling_span in all_filling_spans:
            sub_sub_df = df_with_preds.loc[df_with_preds.topic==topic].loc[df_with_preds.summarySpanOffsets == filling_span]
            extra_row = dict(sub_sub_df.loc[sub_sub_df['pred_prob'].idxmax()]) # find alignment with highest probability
            extra_rows.append(extra_row)
    return extra_rows

def get_filtered_df(df, preds_prob, args):
    positiveAlignments = df[preds_prob >= float(args.filt_threshold)]
    positiveAlignments['pred_prob'] = preds_prob[preds_prob >= args.filt_threshold]

    if args.filler:
        if not args.summaries_len_json:
            print("ERROR: flag '--filler' requires flag '--summaries-len-json /path/to/summaries_len_json_file'")
            exit()
        with open(args.summaries_len_json) as f1:
            summaries_len_dict = json.load(f1)
        extra_df_rows = get_filling_df_rows(summaries_len_dict, positiveAlignments, df, preds_prob)
        positiveAlignments = positiveAlignments.append(extra_df_rows, ignore_index=True)

    positiveAlignments = positiveAlignments[['database', 'topic', 'summaryFile', 'scuSentCharIdx', 'scuSentence',
                                             'documentFile', 'docSentCharIdx', 'docSentText', 'docSpanOffsets',
                                             'summarySpanOffsets', 'docSpanText', 'summarySpanText', 'Quality']]
    return positiveAlignments



def save_df_with_preds(df, preds_prob, csv_path):
    df_with_preds = df.copy()
    df_with_preds['pred_prob'] = preds_prob
    df_with_preds.to_csv(os.path.join(csv_path,'dev_with_probs.csv'), index=False)

def calc_final_alignments(csv_path, model_path, preds, preds_prob, args):
    df = pd.read_csv(os.path.join(csv_path,'dev.tsv'), sep='\t')
    save_df_with_preds(df, preds_prob, csv_path)

    # positiveAlignments = df[preds==1]

    # positiveAlignments = positiveAlignments[['database', 'topic','summaryFile', 'scuSentCharIdx', 'scuSentence',
    #                                          'documentFile',	'docSentCharIdx', 'docSentText', 'docSpanOffsets',
    #                                          'summarySpanOffsets', 'docSpanText', 'summarySpanText','Quality']]
    # positiveAlignments['pred_prob'] = preds_prob[preds==1]

    positiveAlignments = get_filtered_df(df, preds_prob, args)
    #pred_file_name = csv_path[:-1].split('/')[-1] + '_' + model_path[:-1].split('/')[-1] + '.csv'
    pred_file_name = csv_path[:-1].split('\\')[-1] + '_' + model_path[:-1].split('\\')[-1] + '.csv'
    pred_out_path = os.path.join(csv_path, pred_file_name)
    positiveAlignments.to_csv(pred_out_path, index=False)




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


