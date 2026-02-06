import logging
import wandb
import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import copy
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Union, List, Dict, Tuple
from fewuser.trainers import CLTrainer
from fewuser.models import Locator
from tools.label_helper import *
from tools.data_helper import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # a6000
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    set_seed(training_args.seed)

    training_args.output_dir = os.path.join(training_args.output_dir, model_args.model_name_or_path)
    model_args.templates = [model_args.templates[idx] for idx in model_args.templates_idx]
    training_args.logging_steps, training_args.save_steps = training_args.eval_steps, training_args.eval_steps
    if data_args.dataset_name=='Twitter':
        training_args.metrics_eval += ['mean_dist', 'median_dist']
        model_args.templates = [t.replace('Flickr', 'Twitter') if 'Flickr' in t else t for t in model_args.templates]

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
                   f"gpu name: {torch.cuda.get_device_name(torch.cuda.current_device())}, distributed training: {bool(training_args.local_rank != -1)}, "
                   f"16-bits training: {training_args.fp16} ")
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    if data_args.input_type.split('-')[1] in ['in1', 'in2']:
        data_args.max_seq_length = 512
    elif data_args.input_type.split('-')[1] in ['inN', 'inuser+1']:
        data_args.max_seq_length = 256
    elif data_args.input_type.split('-')[1] in ['inuser+N', 'noin']:
        data_args.max_seq_length = 128

    data, label_dict, label_latlng_dict = load_and_process_data(data_args.dataset_name, '', data_args.min_label_counts, data_args.max_shot, data_args.split_ratios, training_args.do_fsl, training_args.do_train, training_args.num_run, training_args.num_shot, training_args.fsl_random_seed)
    model_args.classnames, data_args.num_classes = list(label_dict.keys()), len(label_dict)
    cols_dict = get_columns(data_args.dataset_name)
    data_dict, data_args.columns, cate_num_classes = build_geo_dataset_dict(data, cols_dict, data_args.num_tweets, model_args.model_name_or_path, data_args.max_seq_length, label_dict, data_args.dataset_name, data_args.input_type)

    wandb_config = copy.deepcopy(data_args)
    for args in [training_args, model_args]:
        for key, value in vars(args).items(): setattr(wandb_config, key, value)
    wandb.init(project=training_args.wandb_project, name=f"fewuser", config=wandb_config)

    num_feats = calculate_num_user_feats(data_args.input_type, data_args.num_tweets, cols_dict)

    model = Locator(model_args, cate_num_classes, data_args.input_type, data_args.dataset_name, num_feats)

    result_val_list, result_test_list = [], []
    for run_index in range(training_args.num_run):
        logger.info('-'*50+f'Start {run_index+1}th run now, {training_args.num_run-run_index-1} left.'+'-'*50)
        train_examples = data_dict[f'train_fsl_{run_index}'] if training_args.do_fsl else data_dict['train'] if training_args.do_train else None
        trainer = CLTrainer(model=model, args=training_args, train_dataset=train_examples, eval_dataset=data_dict['val'], tokenizer=model.tokenizer,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.earlystop_patience)])
        trainer.model_args = model_args
        if training_args.do_train: trainer.train()
        result_val_list.append(trainer.evaluate(eval_dataset=data_dict['val'], metric_key_prefix='val', metrics=training_args.metrics_eval, label_latlng_dict=label_latlng_dict))
        result_test_list.append(trainer.evaluate(eval_dataset=data_dict['test'], metric_key_prefix='test', metrics=training_args.metrics_eval, label_latlng_dict=label_latlng_dict))

    results_val_avg, results_test_avg = {}, {}
    for metric in training_args.metrics_eval:
        temp_val = [r[f'val_{metric}'] for r in result_val_list]
        results_val_avg[f'val_{metric}_list'] = temp_val
        results_val_avg[f'val_{metric}_avg'] = np.average(temp_val)

        temp_test = [r[f'test_{metric}'] for r in result_test_list]
        results_test_avg[f'test_{metric}_list'] = temp_test
        results_test_avg[f'test_{metric}_avg'] = np.average(temp_test)

    for k, v in results_val_avg.items(): wandb.log({k: v})
    for k, v in results_test_avg.items(): wandb.log({k: v})

    logger.info("Average results on val set %s", {k: '%.4f' % v for k, v in results_val_avg.items() if isinstance(v, float)})
    logger.info("Average results on test set %s", {k: '%.4f' % v for k, v in results_test_avg.items() if isinstance(v, float)})

if __name__ == "__main__":
    # model args
    @dataclass
    class ModelArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
        """
        # Huggingface's original arguments
        model_name_or_path: Optional[str] = field(
            default='princeton-nlp/sup-simcse-roberta-large',
            metadata={
                "help": "Implemented options: bert-base-cased, bert-base-uncased, gpt2, roberta-base, vinai/bertweet-large, "
                        "allenai/longformer-base-4096，princeton-nlp/sup-simcse-roberta-base， princeton-nlp/sup-simcse-bert-base-uncased,"
                        "albert-base-v2, SpanBERT/spanbert-large-cased, distilbert-base-uncased"
                        "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            },
        )
        cache_dir: Optional[str] = field(
            default=None,
            metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
        )
        use_fast_tokenizer: bool = field(
            default=True,
            metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
        )
        model_revision: str = field(
            default="main",
            metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
        )
        use_auth_token: bool = field(
            default=False,
            metadata={
                "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                        "with private models)."
            },
        )
        use_dual_encoder: bool = field(
            default=False,
            metadata={
                "help": "default not share encoders weights, then use two individual encoders for post and poi"
            }
        )
        temp: float = field(
            default=0.03,
            metadata={
                "help": "Temperature for softmax."
            }
        )
        pooler_type: str = field(
            default="cls_before_pooler",
            metadata={
                "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
            }
        )
        transformer_after_pooler: Optional[bool] = field(
            default=False,
            metadata={
                'help': 'if apply one layer of transformer over the pooler output of bakcbone (bert)'
            }
        )
        transformer_after_pooler_nhead: Optional[int] = field(
            default=4,
            metadata={
                'help': 'if apply one layer of transformer over the pooler output of bakcbone (bert)'
            }
        )
        transformer_after_pooler_nlayer: Optional[int] = field(
            default=1,
            metadata={
                'help': 'if apply one layer of transformer over the pooler output of bakcbone (bert)'
            }
        )
        user_encoder_type: Optional[str] = field(
            default='stack-adapter',
            metadata={
                'help': '{stack, concat}-{lstm, bilstm, rnn, gru, transformer, adapter, mllp, ap}, ap only applies to stack'
            }
        )
        user_encoder_reduction_factor: Optional[int] = field(
            default=16,
            metadata={
                'help': 'reduction factor for adapter'
            }
        )
        user_encoder_dropout: Optional[float] = field(
            default=0.2,
            metadata={
                'help': 'dropout rate for adapter'
            }
        )
        apply_mlp_after_user_encoder: Optional[bool] = field(
            default=False,
            metadata={'help': ''}
        )
        user_encoder_nlayer: Optional[int] = field(
            default=2,
            metadata={
                'help': 'if apply one layer of transformer over the pooler output of bakcbone (bert)'
            }
        )
        user_encoder_nhead: Optional[int] = field(
            default=4,
            metadata={
                'help': 'if apply one layer of transformer over the pooler output of bakcbone (bert)'
            }
        )
        pooler_type_gpt: str = field(
            default="avg",
            metadata={
                "help": "What kind of pooler to use ('last_token', 'last_unpad_token', 'avg', 'avg_unpad_tokens')."
            }
        )
        hard_negative_type: str = field(
            default="multi-nominal",
            metadata={
                "help": "how to sample/get the negative label, multi-nominal, top."
            }
        )
        hard_negative_k: int = field(
            default=6,
            metadata={
                "help": "the number of negative labels, best 7."
            }
        )
        num_head_of_attention: int = field(
            default=64,
            metadata={
                "help": "the num head of cross attention, which is applied to fuse text embedding and label embedding for loss tlm"}
        )
        fusion_type: str = field(
            default="cls_concat_adapter",
            metadata={
                "help": "cls_ca, cls_sum_mha, cls_sum_adapter, cls_sum_mlp, cls_sum_transformer, "
                        "cls_concat_mha, cls_concat_adapter, cls_concat_mlp, cls_concat_transformer, "
                        "pooler_cls_ca, pooler_cls_sum_mha, pooler_cls_sum_adapter, pooler_cls_sum_mlp, pooler_cls_sum_transformer, "
                        "pooler_cls_concat_mha, pooler_cls_concat_adapter, pooler_cls_concat_mlp, pooler_cls_concat_transformer, "
            }
        )
        mlp_after_fusion: bool = field(
            default=False,
            metadata={
                "help": "if apply an mlp layer before tlm loss"
            }
        )
        log_loss: bool = field(
            default=False,
            metadata={
                "help": "whether to do log transform of loss_tlm and loss_tlm"
            },
        )
        pad_to_max_length: bool = field(
            default=True,
            metadata={
                "help": "Whether to pad all samples to `max_seq_length`. "
                        "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            },
        )
        classnames: List[str] = field(
            default_factory=lambda: [None],
            metadata={
                "help": "classnames"
            }
        )
        template_with_learnable_tokens: bool = field(
            default=False,
            metadata={'help': 'use learnable tokens for template or use fixed manual template'}
        )
        init_manual_template: bool = field(
            default=False,
            metadata={'help': 'if learnable tokens are initialized from manual templates.'}
        )
        num_learnable_tokens: int = field(
            default=6,
            metadata={'help': 'number of learnable tokens used in template if not init from manual templates.'}
        )
        templates_idx: List[int] = field(
            default_factory=lambda: [8],
            metadata={
                "help": "help to select templates from the list below."
            }
        )
        templates: List[str] = field(
            default_factory=lambda: [
                f"[CLASS]",
                f"A user from the city [CLASS] .",
                f"Someone from the city [CLASS] .",
                f"Question: where does this Flickr user reside in? Answer: [CLASS] .",
                f"This Flickr user resides in [CLASS] .",
                f"[CLASS] .",
                f"Question: which city does this Flickr user live in? Answer: [CLASS] .",
                f"Question: which city does this Flickr user reside in? Answer: [CLASS] .",
                f"Question: which city does this user live in? Answer: [CLASS] .",
                f"Question: which city does this user reside in? Answer: [CLASS] .",
                f"Question: where does this user reside in? Answer: [CLASS] .",
                f"Question: what city is this Flickr user from? Answer: [CLASS] .",
                f"Question: what city is this user from? Answer: [CLASS] .",
                f"Question: where is this Flickr user from? Answer: [CLASS] .",
                f"I'm in [CLASS] .",
                f"I really like my city [CLASS] .",
                f"A Flickr user from the city [CLASS] .",
                f"A Flickr user resides in [CLASS] .",
                f"A user resides in [CLASS] .",
                f"A Flickr user resides in the city [CLASS] .",
                f"A user resides in the city [CLASS] .",
                f"A guy from the city [CLASS] .",
                f"[CLASS] 's own.",
                f"Hailing from [CLASS] .",
                f"A local from [CLASS] .",
                f"[CLASS]  in the house!",
                f"Representing [CLASS] .",
                f"This is [CLASS]  calling.",
                f"This user resides in the city [CLASS] .",
            ],
            metadata={
                'help': 'templates used to extend the classnames into sentences, 27'
            }
        )
        smooth_loss: bool = field(
            default=False,
            metadata={
                'help': 'if apply label smoothing to crossentropy loss or not'
            }
        )
        use_tlm_loss: bool = field(
            default=False,
            metadata={
                'help': 'if use tlm (text label matching) loss'
            }
        )
        use_tlc_loss: bool = field(
            default=True,
            metadata={
                'help': 'if use tlc (text label contrastive) loss'
            }
        )
        freeze_encoder: bool = field(
            default=False,
            metadata={
                'help': 'if freeze encoder (for user) or not'
            }
        )
        freeze_label_encoder: bool = field(
            default=False,
            metadata={
                'help': 'if freeze encoder (for location) or not'
            }
        )

    # data args
    @dataclass
    class DataTrainingArguments:
        """
        Arguments pertaining to what data we are going to input our model for training and eval.
        """
        min_label_counts: Optional[int] = field(
            default=40,
            metadata={"help": ""},
        )
        input_type: str = field(
            default='all-in1-text',
            metadata={
                "help": "{all, noposttime, nopostmeta}-{in1, in2, inN, inuser+1, inuser+N, noin}-{text, text+cate}, "
                        "all-in1-text, all-in2-text, all-inN-text, all-inuser+1-text, all-inuser+N-text, all-noin-text, all-inuser+1-text+cate, all-inuser+N-text+cate, all-noin-text+cate"
                        "noposttime-in1-text, noposttime-in2-text, noposttime-inN-text, noposttime-inuser+1-text, noposttime-inuser+N-text, noposttime-noin-text, noposttime-inuser+1-text+cate, noposttime-inuser+N-text+cate, noposttime-noin-text+cate"
                        "nopostmeta-in1-text, nopostmeta-in2-text, nopostmeta-inN-text, nopostmeta-inuser+1-text, nopostmeta-inuser+N-text, nopostmeta-noin-text, nopostmeta-inuser+1-text+cate, nopostmeta-inuser+N-text+cate, nopostmeta-noin-text+cate"
                }
        )
        num_tweets: int = field(
            default=4, metadata={
                "help": ""}
        )
        max_shot: int = field(
            default=16,
            metadata={"help": ""},
        )
        dataset_name: str = field(
            default='Twitter', metadata={"help": "Twitter, Flickr"}
        )
        columns: List[int] = field(
            default_factory=lambda: ['merge_text'],
            metadata={'help': "['merge_user', 'merge_tweet'], ['merge_text']"}
        )
        num_classes: int = field(
            default=None,
            metadata={'help': ''}
        )
        split_ratios: List[float] = field(
            default_factory=lambda: [0.7, 0.85, 0.15],
            metadata={'help': 'train, val, test, split ratio'}
        )
        overwrite_cache: bool = field(
            default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
        )
        preprocessing_num_workers: Optional[int] = field(
            default=None,
            metadata={"help": "The number of processes to use for the preprocessing."},
        )
        max_seq_length: Optional[int] = field(
            default=512,
            metadata={
                "help": "The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated."
            },
        )


    # training args
    @dataclass
    class OurTrainingArguments(TrainingArguments):
        output_dir: Optional[str] = field(
            default='result/fewuser/',
            metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
        )
        do_train: bool = field(
            default=False,
            metadata={"help": ""},
        )
        do_fsl: bool = field(
            default=False,
            metadata={'help': 'do few-shot learning or not, means do sampling or not before training'}
        )
        num_shot: int = field(
            default=1,
            metadata={"help": "if do few-shot learning, select num_shot samples from each class randomly"},
        )
        num_run: int = field(
            default=1,
            metadata={"help": ""},
        )
        fsl_random_seed: List[int] = field(
            default_factory=lambda: [2731, 5766, 6245, 8865, 9471],
            metadata={
                "help": "if random_seed_provided is True, fsl_random_seed must be set with num_run numbers, [10, 50, 100, 2731, 1844], [5140, 5766, 6245, 9471, 8865]"},
        )
        seed: int = field(
            default=777,
            metadata={"help": ""},
        )
        evaluation_strategy: Optional[str] = field(
            default='steps',
            metadata={"help": ""},
        )
        eval_steps: Optional[float] = field(
            default=10,
            metadata={"help": ""},
        )
        save_steps: Optional[float] = field(
            default=10,
            metadata={"help": ""},
        )
        logging_steps: Optional[float] = field(
            default=160,
            metadata={"help": ""},
        )
        earlystop_patience: int = field(
            default=10,
            metadata={"help": "only applicable when set use_early_stopping as True"},
        )
        label_data: Optional[str] = field(
            default=None,
            metadata={"help": ""}
        )
        fp16: bool = field(
            default=True,
            metadata={"help": ""},
        )
        load_best_model_at_end: bool = field(
            default=True,
            metadata={"help": ""},
        )
        save_total_limit: int = field(
            default=3,
            metadata={'help': ''},
        )
        metric_for_best_model: str = field(
            default='eval_top1',
            metadata={
                "help": ""
            }
        )
        metrics_eval: List[str] = field(
            default_factory=lambda: ['top1', 'f1_micro', 'f1_macro', 'f1_weight'],
            metadata={'help': 'metrics used for evaluation'}
        )
        overwrite_output_dir: bool = field(
            default=True,
            metadata={"help": ""},
        )
        num_train_epochs: float = field(
            default=100,
            metadata={
                "help": ""
            }
        )
        per_device_train_batch_size: int = field(
            default=8,
            metadata={
                "help": ""
            }
        )
        per_device_eval_batch_size: int = field(
            default=4,
            metadata={
                "help": ""
            }
        )
        learning_rate: float = field(
            default=8e-6,
            metadata={
                "help": "8e-6, 2e-5, 5.5e-6"
            }
        )
        adam_beta1: Optional[float] = field(
            default=0.9,
            metadata={
                "help": ""
            }
        )
        adam_beta2: Optional[float] = field(
            default=0.999,
            metadata={
                "help": ""
            }
        )
        lr_scheduler_type: Optional[str] = field(
            default='cosine',
            metadata={
                "help": "['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], "
                        "constant and constant with warmup both provide poor performance, cosine with restrats works strangely, only linear and cosine work fine"
            }
        )
        warmup_type: Optional[str] = field(
            default='other',
            metadata={
                "help": "cosine_hard_start_warmup only applicable for cosine scheduler"
            }
        )
        warmup_steps: Optional[float] = field(
            default=0,
            metadata={
                "help": ""
            }
        )
        warmup_prop: Optional[float] = field(
            default=0.0,
            metadata={
                "help": "if warmup_prop and warmup_steps are both not null, use prop"
            }
        )
        warmup_min_lr: Optional[float] = field(
            default=1e-8,
            metadata={
                "help": "must be provided if using cosine_with_hard_start_and_warmup"
            }
        )
        wandb_project: Optional[str] = field(
            default='fewuser-feb',
            metadata={
                "help": ""
            }
        )
        num_workers: int = field(
            default=None
        )

    main()
