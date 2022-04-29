from utils import *
from docSum2MRPC_Aligner import docSum2MRPC_Aligner
sys.path.append(f'{os.path.join(os.getcwd(), "transformers", "examples")}')
import pandas as pd

import run_glue
import json
import contextlib
@contextlib.contextmanager
def redirect_argv(num):
    sys._argv = sys.argv[:]
    sys.argv = str(num).split()
    yield
    sys.argv = sys._argv






parser = argparse.ArgumentParser()
parser.add_argument('-data_path', type=str, default='/home/nlp/ernstor1/DUC2004/')  # 'data/final_data/data')
parser.add_argument('-mode', type=str, default='dev')
parser.add_argument('-log_file', type=str, default='results/dev_log.txt')
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-alignment_model_path', type=str, required=True)
parser.add_argument('-database', type=str, default='None')

# added for the controlled reduction project
parser.add_argument("--filt-threshold", type=float, default=0.5, help="threshold for filtering alignments (if the probability of the alignment is greater or equal to the threshold). in fraction (e.g., --filt-threshold 0.1 for threshold of 10%)")
parser.add_argument("--filler", action="store_true", default=False, help="flag for filling in spans of the summary that weren't filled (using the maximum probabilty covering that span, even if below threshold). Requires to pass also json file with lengths of the summaries")
parser.add_argument("--spans-doc", type=str, default=None, help="if already divided into spans (and no need for the span division part, just the alignment). In csv form. Also, make sure that of same format as the dev.tsv file - see method \'save_predictions\' in class \'docSum2MRPC_Aligner\'")
args = parser.parse_args()


# added for the controlled reduction project - extra flags for run_glue
filt_threshold = args.filt_threshold
filler = args.filler
extra_flags = f" --filt-threshold {filt_threshold}"  if float(filt_threshold)>0 else ""
if filler:
    extra_flags+=f" --filler"


summary_files = glob.glob(f"{args.data_path}/summaries/*")
sfile_lens = dict()
sfile_lens['summaries lens'] = dict()

if not args.spans_doc:
    aligner = docSum2MRPC_Aligner(data_path=args.data_path, mode=args.mode,
                     log_file=args.log_file, output_file = args.output_path,
                     database=args.database)
    logging.info(f'output_file_name: {args.output_path}')

    for sfile in summary_files:
            print ('Starting with summary {}'.format(sfile))
            aligner.read_and_split(args.database, sfile)
            aligner.scu_span_aligner()
            with open(sfile, 'r', encoding='utf-8') as f1:
                sfile_lens['summaries lens'][f'{os.path.basename(sfile)}'] = len(f1.read())
    aligner.save_predictions()

else:
    spans_df = pd.read_csv(args.spans_doc)
    spans_df.to_csv(os.path.join(args.output_path, 'dev.tsv'), index=False, sep='\t')
    for sfile in summary_files:
            with open(sfile, 'r', encoding='utf-8') as f1:
                sfile_lens['summaries lens'][f'{os.path.basename(sfile)}'] = len(f1.read())


json_object = json.dumps(sfile_lens)
summaries_len_file = os.path.join(args.data_path, 'summaries_lens.json')
with open(f"{summaries_len_file}", "w") as outfile:
    outfile.write(json_object)





with redirect_argv('python --model_type roberta --model_name_or_path roberta-large-mnli --task_name MRPC --do_eval'
                           f' --calc_final_alignments --weight_decay 0.1 --data_dir {args.output_path}'
                           ' --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 2e-6'
                           ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
                           f' --output_dir {args.alignment_model_path}'
                           f'{extra_flags}'
                           f' --summaries-len-json {summaries_len_file}'):
        run_glue.main()





