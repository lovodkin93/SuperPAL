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
parser.add_argument('--data_path', type=str, default='../train_QASEM_data')
parser.add_argument('--outdir', type=str, default='../finetuned_model')  # 'data/final_data/data')

args = parser.parse_args()

batch_size = 16

with redirect_argv('python --model_type roberta --model_name_or_path roberta-large-mnli --task_name MRPC --do_eval --do_train'
                   f' --weight_decay 0.1 --data_dir {args.data_path}'
                   f' --max_seq_length 128 --per_gpu_train_batch_size {batch_size} --per_gpu_eval_batch_size {batch_size} --learning_rate 2e-6'
                   ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
                   f' --output_dir {args.outdir}'):
    run_glue.main()





