from transformers.utils import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.optimization import Adafactor, AdamW, get_scheduler
import wandb
from tqdm import tqdm
from tools.metric_helper import *
from tools.scheduler_helper import *
from transformers import Trainer, AdamW
logger = logging.get_logger(__name__)

class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metrics: Optional[List[str]] = ['top1'], # top1, f1_micro, f1_macro, f1_weight
        metric_key_prefix: str = "eval",
        label_latlng_dict: Optional[Dict] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        def batcher(batch_size, eval_dataloader, metrics, label_latlng_dict):
            params = {
                'top1': AverageMeter("Acc@1", ":.4f"),
                'f1_micro': AverageMeter("F1_Micro", ":.4f"),
                'f1_macro': AverageMeter("F1_Macro", ":.4f"),
                'f1_weight': AverageMeter("F1_Weight", ":.4f"),
                'mean_dist': AverageMeter("Mean_Dist", ":.4f"),
                'median_dist': AverageMeter("Median_Dist", ":.4f"),
            }
            dists = []
            with torch.no_grad():
                for inputs in tqdm(eval_dataloader):
                    for k in inputs: inputs[k] = inputs[k].to(self.args.device)
                    logits = self.model(**inputs, inference=True)
                    labels = inputs['labels']
                    params['top1'].update(getAccuracy(logits, labels, batch_size, topk=(1,))[0])
                    if len(metrics)>1:
                        f1_mi, f1_ma, f1_we = getF1score(logits, labels, avg=('micro', 'macro', 'weighted'))
                        params['f1_micro'].update(f1_mi)
                        params['f1_macro'].update(f1_ma)
                        params['f1_weight'].update(f1_we)
                    if len(metrics)>4:
                        dists.extend(get_distance_list(logits, labels, label_latlng_dict))
            if len(dists)>0:
                params['mean_dist'].update(np.mean(dists)/1000)
                params['median_dist'].update(np.median(dists)/1000)
            return params

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        batch_size = self.args.per_device_eval_batch_size
        params = batcher(batch_size, eval_dataloader, metrics, label_latlng_dict)

        metrics_keys_with_prefix = [f'{metric_key_prefix}_{k}' for k in metrics]
        metrics_values = [params[k].avg.item() for k in metrics]
        metrics_result = {k: v for k, v in zip(metrics_keys_with_prefix, metrics_values)}
        metrics_dict = {k: '%.4f' % v for k, v in zip(metrics_keys_with_prefix, metrics_values)}

        logger.info(metrics_dict)
        for k, v in zip(metrics_keys_with_prefix, metrics_values): wandb.log({k: v})
        if metric_key_prefix=='eval':  self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_result)
        return metrics_result

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps)

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters()],
                                             "weight_decay": self.args.weight_decay}]
            optimizer_kwargs = {"betas": (self.args.adam_beta1, self.args.adam_beta2),
                                "eps": self.args.adam_epsilon,
                                "lr": self.args.learning_rate}
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int):
        """
        当warmup steps和warmup prop都不是None时，会优先使用prop，二者有着同样的功能
        当lr scheduler type=cosine以及warmup type=cosine hard start warmup时，需要设置warmup steps/prop以及warmup min lr
        其他情况下都只会用到warmup steps
        当lr_scheduler_type in ['linear', 'cosine_with_restarts', 'constant', 'constant_with_warmup']时，只需要设置warmup steps/prop
        当lr scheduler type=cosine时，
        """
        warmup_steps = 0
        if self.args.warmup_type is not None:
            if self.args.warmup_prop is not None: warmup_steps = int(num_training_steps*self.args.warmup_prop)
            elif self.args.warmup_steps is not None: warmup_steps = self.args.warmup_steps

        if self.lr_scheduler is None:
            if self.args.warmup_type=='cosine_hard_start_warmup' and self.args.lr_scheduler_type=='cosine':
                self.lr_scheduler = WarmupCosineAnnealingLR(self.optimizer, num_training_steps, warmup_lrs=self.args.warmup_min_lr, warmup_epochs=warmup_steps)
            else:
                self.lr_scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=self.optimizer,
                                                  num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        return self.lr_scheduler