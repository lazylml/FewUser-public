import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import RobertaTokenizer
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AutoModel, AutoTokenizer, AlbertPreTrainedModel, DistilBertPreTrainedModel
from transformers import GPT2PreTrainedModel, GPT2Model, AutoConfig
import wandb
from transformers import LongformerTokenizer, LongformerModel
import logging
logger = logging.getLogger(__name__)

MAX_NUM_VECTORS = 15
CLASS_TOKEN = '[CLASS]'

def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class User_Encoder(nn.Module):
    def __init__(self, encoder_type, hidden_size, num_layers, num_heads, apply_mlp_after_user_encoder, num_feats, reduction_factor, dropout):
        super().__init__()
        self.etype0, self.etype1 = encoder_type.split('-')
        self.apply_linear = apply_mlp_after_user_encoder
        assert self.etype0 in ['stack', 'concat']
        assert self.etype1 in ['lstm', 'bilstm', 'rnn', 'gru', 'transformer', 'adapter', 'mlp', 'ap']

        if self.etype0=='stack':
            self.hidden_dims = hidden_size
        else:
            self.hidden_dims = hidden_size*num_feats

        if self.etype1=='lstm':
            self.encoder = nn.LSTM(self.hidden_dims, self.hidden_dims, num_layers)
        elif self.etype1=='bilstm':
            self.encoder = nn.LSTM(self.hidden_dims, self.hidden_dims, num_layers, bidirectional=True)
            self.linear_bi = nn.Sequential(nn.Linear(in_features=self.hidden_dims*2, out_features=self.hidden_dims), nn.ReLU())
        elif self.etype1=='rnn':
            self.encoder = nn.RNN(self.hidden_dims, self.hidden_dims, num_layers)
        elif self.etype1=='gru':
            self.encoder = nn.GRU(self.hidden_dims, self.hidden_dims, num_layers)
        elif self.etype1=='transformer':
            trans_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dims, nhead=num_heads)
            self.encoder = nn.TransformerEncoder(trans_layer, num_layers=num_layers)
        elif self.etype1=='adapter':
            self.encoder = Adapter(self.hidden_dims, reduction_factor=reduction_factor, dropout=dropout)
        elif self.etype1=='mlp':
            self.encoder = nn.Sequential(nn.Linear(in_features=self.hidden_dims, out_features=self.hidden_dims), nn.ReLU())

        if self.apply_linear:
            self.linear_add = nn.Sequential(nn.Linear(in_features=self.hidden_dims, out_features=self.hidden_dims), nn.ReLU())

        if self.etype0=='concat':
            self.linear_keepdim = nn.Sequential(nn.Linear(in_features=self.hidden_dims, out_features=hidden_size), nn.ReLU())

    def forward(self, features_):
        if self.etype0=='stack':
            features = torch.stack(features_)
        else:
            features = torch.cat(features_, dim=1)

        if self.etype1 in ['lstm', 'rnn', 'gru']:
            feat_fused, _ = self.encoder(features)
        elif self.etype1 in ['transformer', 'mlp']:
            feat_fused = self.encoder(features)
        elif self.etype1 == 'bilstm':
            feat_fused, _ = self.encoder(features)
            feat_fused = self.linear_bi(feat_fused)
        elif self.etype1 == 'adapter':
            feat_fused = self.encoder(features, features)
        elif self.etype1 == 'ap':
            feat_fused = features

        if self.apply_linear:
            feat_fused = self.linear_add(feat_fused)

        if self.etype0=='stack':
            output = torch.mean(feat_fused, dim=0)
        else:
            output = self.linear_keepdim(feat_fused)

        return output

class Transformer(nn.Module):
    def __init__(self, hidden_size, nhead, nlayer):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=nlayer)
    def forward(self, feat):
        return self.transformer(feat.view(-1, 1, self.hidden_size)).view(-1, self.hidden_size)

class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class Pooler_gpt(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['last_token', 'last_unpad_token', 'avg', 'avg_unpad_tokens'], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, outputs, batch_size, input_ids, pad_token_id):
        last_hidden = outputs.last_hidden_state

        if self.pooler_type == 'last_token':
            pooler_output = last_hidden[:, -1]
            return pooler_output
        elif self.pooler_type == 'avg':
            pooler_output = torch.mean(last_hidden, dim=1)
            return pooler_output
        elif self.pooler_type == 'last_unpad_token':
             if pad_token_id is None:
                 sequence_lengths = -1
             else:
                 if input_ids is not None:
                     sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(input_ids.device)
                 else:
                     sequence_lengths = -1
             pooler_output = last_hidden[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
             return pooler_output
        elif self.pooler_type == 'avg_unpad_tokens':
            if pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(input_ids.device)
                else:
                    sequence_lengths = -1
            mask = torch.arange(last_hidden.size(1)).unsqueeze(0).to(last_hidden.device) < sequence_lengths.unsqueeze(1)
            masked_last_hidden = last_hidden * mask.unsqueeze(-1)
            sum_last_hidden = masked_last_hidden.sum(dim=1)
            pooler_output = sum_last_hidden / sequence_lengths.unsqueeze(1).float()
            return pooler_output
        else:
            raise NotImplementedError

class Adapter(nn.Module):
    """
    Inter-view Adapter
    """
    def __init__(self, in_dim, reduction_factor=16, dropout=0):
        super().__init__()
        self.mid_dim = in_dim//reduction_factor
        self.in_dim = in_dim
        self.mode='interview'
        self.layer_norm = nn.LayerNorm(in_dim)
        self.batch_norm = nn.BatchNorm1d(in_dim)
        self.linear_down = nn.Linear(in_features=in_dim, out_features=self.mid_dim)
        self.linear_up = nn.Linear(in_features=self.mid_dim, out_features=in_dim)
        self.linear_in = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.linear_mid = nn.Linear(in_features=self.mid_dim, out_features=self.mid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, residual_feat):
        # output = feat.reshape(batch_size, -1, self.in_dim)
        output = feat
        if self.mode=='interview':
            output = self.layer_norm(output)
            output = self.dropout(output)
            output = self.linear_down(output)
            output = self.relu(output)
            output = self.dropout(output)
            output = self.linear_mid(output)
            output = self.relu(output)
            output = self.linear_up(output)
            output = self.relu(output)
            output = output + residual_feat
        elif self.mode=='speech':
            # 这里还有一个额外的处理层，看下效果再决定要不要添加
            output = self.layer_norm(output)
            output = self.linear_down(output)
            output = self.relu(output)
            output = self.linear_mid(output)
            output = self.linear_up(output)
            output = output+residual_feat
        else:
            raise TypeError(f'the given adapter mode {self.mode} is not implemented yet.')

        return output

class Loss_tlm(nn.Module):
    def __init__(self, hidden_size, args):
        super().__init__()

        self.mlp_after_fusion = args.mlp_after_fusion
        self.num_classes = len(args.classnames)
        self.num_prompts = len(args.templates)
        self.hidden_size = hidden_size
        self.fusion_type = args.fusion_type
        self.num_head = args.num_head_of_attention
        self.hard_negative_type = args.hard_negative_type
        self.hard_negative_k = args.hard_negative_k

        if '_ca' in self.fusion_type:
            self.attention = nn.MultiheadAttention(self.hidden_size, self.num_head, batch_first=True)
        elif '_mha' in self.fusion_type:
            self.attention = nn.MultiheadAttention(self.hidden_size*2 if 'concat_' in self.fusion_type else self.hidden_size, self.num_head, batch_first=True)
        if '_adapter' in self.fusion_type:
            self.adapter = Adapter(self.hidden_size*2 if 'concat_' in self.fusion_type else self.hidden_size)
        if '_transformer' in self.fusion_type:
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        if '_mlp' in self.fusion_type:
            self.mlp = nn.Sequential(nn.Linear(in_features=self.hidden_size*2 if 'concat_' in self.fusion_type else self.hidden_size,
                                               out_features=self.hidden_size*2 if 'concat_' in self.fusion_type else self.hidden_size), nn.ReLU())

        if 'concat_' in self.fusion_type:
            self.mlp1 = nn.Sequential(nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size), nn.ReLU())

        if self.mlp_after_fusion:
            self.mlp2 = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU())

        self.loss_head = nn.Linear(self.hidden_size, 2)

    def get_weights(self, sim_score, labels, batch_size):
        with torch.no_grad():
            weights_t2l = torch.stack([F.softmax(sim_score[:, :, idx_pmt], dim=1) for idx_pmt in range(self.num_prompts)])
            for score in weights_t2l:
                for i, label in enumerate(labels):
                    score[i, label.item()] = 0
        if self.hard_negative_type=='multi-nominal':
            # shape(1, 8), (num_prompts, batch size)
            idx_labels_neg = torch.stack([torch.stack([torch.multinomial(weights_t2l[idx_pmt,b], self.hard_negative_k) for b in range(batch_size)]) for idx_pmt in range(self.num_prompts)])
        elif self.hard_negative_type=='top':
            idx_labels_neg = torch.stack([torch.stack([weights_t2l[idx_pmt, b].topk(self.hard_negative_k)[1] for b in range(batch_size)]) for idx_pmt in range(self.num_prompts)])
        else:
            raise NotImplementedError(f'The given hard_negative_type {self.hard_negative_type} is not in implemented list.')
        return idx_labels_neg

    def get_label_embed(self, inputs_labels, labels, idx_labels_neg, batch_size):
        embeds_labels_pmt = torch.stack([inputs_labels[:, idx_pmt] for idx_pmt in range(self.num_prompts)])  # num prompts, num classes, (sequence length), hidden size
        embeds_labels_pos = torch.stack([torch.stack([embeds_labels_pmt[idx_pmt, label.item()] for label in labels]) for idx_pmt in range(self.num_prompts)])  # num prompts, batch size, (sequence length), hidden size
        if len(idx_labels_neg.shape)==3:
            embeds_labels_negs = []
            for i in range(idx_labels_neg.shape[2]):
                idx_labels = idx_labels_neg[:, :, i]
                embeds_labels = torch.stack([torch.stack([embeds_labels_pmt[idx_pmt, idx_labels[idx_pmt, b]] for b in range(batch_size)]) for idx_pmt in range(self.num_prompts)])
                embeds_labels_negs.append(embeds_labels)
        else:
            embeds_labels_negs = [torch.stack([torch.stack([embeds_labels_pmt[idx_pmt, idx_labels_neg[idx_pmt, b]] for b in range(batch_size)]) for idx_pmt in range(self.num_prompts)])]  # num prompts, batch size, (sequence length), hidden size
        return embeds_labels_pos, embeds_labels_negs

    def get_inputs(self, vec_feat, vec_label):
        inputs = vec_feat # batch size, hidden size
        inputs_labels = vec_label.view(self.num_classes, self.num_prompts, self.hidden_size) # num classes, num prompts, hidden size
        return inputs, inputs_labels

    def get_text_label_embeds(self, embeds_inputs, embeds_labels, batch_size):
        if 'embed_' in self.fusion_type:
            embeds_tl = torch.stack([torch.concat([embeds_inputs, e], dim=1) for e in embeds_labels])  # num prompts, batch size, (sequence length), hidden size
            if '_mha' in self.fusion_type:
                embeds_tl_fused = [self.attention(e, e, e)[0].mean(dim=1) for e in embeds_tl]
            elif '_adapter' in self.fusion_type:
                embeds_tl_fused = [self.adapter(e, e).mean(dim=1) for e in embeds_tl]
            elif '_transformer' in self.fusion_type:
                embeds_tl_fused = [self.transformer(e.view(batch_size, -1, self.hidden_size)).mean(dim=1) for e in embeds_tl]
            elif '_mlp' in self.fusion_type:
                embeds_tl_fused = [self.mlp(e).mean(dim=1) for e in embeds_tl]
            elif self.fusion_type == 'embed_avg':
                embeds_tl_fused = [e.mean(dim=1) for e in embeds_tl]
            else:
                raise NotImplementedError(f'The given fusion_type {self.fusion_type} is not in implemented list.')
        else:
            if '_ca' in self.fusion_type:
                embeds_inputs_reshape = embeds_inputs.view(batch_size, 1, -1)
                embeds_labels_reshape = embeds_labels.view(self.num_prompts, batch_size, 1, -1)
                embeds_tl_fused = [self.attention(embeds_inputs_reshape, e, e)[0].view(batch_size, self.hidden_size) for e in embeds_labels_reshape] # num prompts, batch size, hidden size

            else:
                if 'sum_' in self.fusion_type:
                    embeds_tl = torch.stack([embeds_inputs + e for e in embeds_labels])  # num prompts, batch size, hidden size
                elif 'concat_' in self.fusion_type:
                    embeds_tl = torch.stack([torch.concat([embeds_inputs, e], dim=1) for e in embeds_labels])  # num prompts, batch size, hidden size*2

                if '_mha' in self.fusion_type:
                    embeds_tl_fused = [self.attention(e, e, e)[0] for e in embeds_tl]
                elif '_adapter' in self.fusion_type:
                    embeds_tl_fused = [self.adapter(e, e) for e in embeds_tl]
                elif '_transformer' in self.fusion_type:
                    embeds_tl_fused = [self.transformer(e.view(batch_size, -1, self.hidden_size)).view(batch_size, -1) for e in embeds_tl]
                elif '_mlp' in self.fusion_type:
                    embeds_tl_fused = [self.mlp(e) for e in embeds_tl]
                else:
                    raise NotImplementedError(f'The given fusion_type {self.fusion_type} is not in implemented list.')

                if 'concat_' in self.fusion_type:
                    embeds_tl_fused = [self.mlp1(e) for e in embeds_tl_fused]

        return embeds_tl_fused

    def get_loss(self, embeds_pos, embeds_negs, device, batch_size):
        num_neg_labels = len(embeds_negs)
        embeds_negs = torch.stack([torch.stack(e) for e in embeds_negs])
        embeds_negs = [embeds_negs[:, idx].view(-1, self.hidden_size) for idx in range(self.num_prompts)]
        tl_embeddings = [torch.concat([embeds_pos[idx_pmt], embeds_negs[idx_pmt]], dim=0) for idx_pmt in range(self.num_prompts)]
        tl_outputs = [self.loss_head(tle) for tle in tl_embeddings]
        tlm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size*num_neg_labels, dtype=torch.long)], dim=0).to(device)
        loss_tlm = torch.stack([F.cross_entropy(text_label_output, tlm_labels) for text_label_output in tl_outputs]).mean()
        return loss_tlm

    def forward(self, vec_feat, vec_label, labels, sim_score):
        # 因为前序操作涉及到User部分的多个vectors的融合，所以不能提供embeddings作为输入，所以所有和Embed相关的fusion type都不可用
        batch_size = vec_feat.shape[0]
        idx_labels_neg = self.get_weights(sim_score, labels, batch_size)

        inputs, inputs_labels = self.get_inputs(vec_feat, vec_label)
        embeds_labels_pos, embeds_labels_negs = self.get_label_embed(inputs_labels, labels, idx_labels_neg, batch_size)
        embeds_tl_fused_pos = self.get_text_label_embeds(inputs, embeds_labels_pos, batch_size)
        embeds_tl_fused_negs = [self.get_text_label_embeds(inputs, embeds_labels_neg, batch_size) for embeds_labels_neg in embeds_labels_negs]

        if self.mlp_after_fusion:
            embeds_tl_fused_pos = [self.mlp2(o) for o in embeds_tl_fused_pos]
            embeds_tl_fused_negs = [[self.mlp2(o) for o in embeds_tl_fused_neg] for embeds_tl_fused_neg in embeds_tl_fused_negs]

        loss_tlm = self.get_loss(embeds_tl_fused_pos, embeds_tl_fused_negs, vec_feat.device, batch_size)

        return loss_tlm

def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]'%(vid)

class Locator(nn.Module):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, model_args, cate_num_classes, input_type, dataset_name, num_feats):
        super().__init__()

        self.model_name = model_args.model_name_or_path
        self.use_dual_encoder = model_args.use_dual_encoder
        self.transformer_after_pooler = model_args.transformer_after_pooler
        self.use_tlm_loss = model_args.use_tlm_loss
        self.use_tlc_loss = model_args.use_tlc_loss
        self.smooth_loss = model_args.smooth_loss
        self.log_loss = model_args.log_loss
        self.num_classes = len(model_args.classnames)
        config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        self.encoder = self.load_model(self.model_name, config)
        tokenizer_kwargs = {"cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        if 'gpt' in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            config.vocab_size = len(self.tokenizer)

        self.mlp = MLPLayer(config)
        self.user_encoder = User_Encoder(model_args.user_encoder_type, config.hidden_size, model_args.user_encoder_nlayer, model_args.user_encoder_nhead, model_args.apply_mlp_after_user_encoder, num_feats, model_args.user_encoder_reduction_factor, model_args.user_encoder_dropout)
        if model_args.transformer_after_pooler:
            self.transformer = Transformer(config.hidden_size, model_args.transformer_after_pooler_nhead, model_args.transformer_after_pooler_nlayer)

        if self.use_dual_encoder:
            self.encoder_location = self.load_model(self.model_name, config)
            self.mlp_location = MLPLayer(config)
            if model_args.transformer_after_pooler:
                self.transformer_location = Transformer(config.hidden_size, model_args.transformer_after_pooler_nhead, model_args.transformer_after_pooler_nlayer)

        self.input_locations = self.prepare_classname_sentences(model_args.template_with_learnable_tokens, model_args.init_manual_template, model_args.templates, model_args.num_learnable_tokens, model_args.classnames)

        self.pooler_type = model_args.pooler_type
        self.pooler = Pooler(self.pooler_type)
        if 'gpt' in self.model_name: self.pooler_type_gpt = Pooler_gpt(model_args.pooler_type_gpt)

        self.sim = Similarity(temp=model_args.temp)

        if model_args.use_tlm_loss: self.loss_tlm = Loss_tlm(config.hidden_size, model_args)

        if model_args.freeze_encoder: self.freeze_model(self.encoder)
        if model_args.freeze_label_encoder and self.encoder_location: self.freeze_model(self.encoder_location)

        if input_type.split('-')[2]=='text+cate' and dataset_name=='Twitter':
            assert len(cate_num_classes)==2
            cate_num_classes_list = list(cate_num_classes.values())
            self.cate_embedding0 = nn.Embedding(cate_num_classes_list[0], self.encoder.config.hidden_size)
            self.cate_embedding1 = nn.Embedding(cate_num_classes_list[1], self.encoder.config.hidden_size)

    def prepare_classname_sentences(self, template_with_learnable_tokens, init_manual_template, templates, num_learnable_tokens, classnames):
        if template_with_learnable_tokens:
            self.prepare_for_dense_prompt()
            # todo revise for multiple templates
            template = self.init_template(init_manual_template, templates[0], num_learnable_tokens)
            classname_sentences = [template.replace(CLASS_TOKEN, cl) for cl in classnames]
            classname_sentences_tokenized = self.tokenizer.batch_encode_plus(classname_sentences, return_tensors='pt', padding=True)
        else:
            template = templates[0]
            classname_sentences = [template.replace(CLASS_TOKEN, cl) for cl in classnames]
            classname_sentences_tokenized = self.tokenizer.batch_encode_plus(classname_sentences, return_tensors='pt', padding=True)
        wandb.config.template = template
        logger.info('Template: %s' % template)
        return classname_sentences_tokenized

    def prepare_for_dense_prompt(self):
        # add new tokens
        new_tokens = [get_new_token(i + 1) for i in range(MAX_NUM_VECTORS)]
        self.tokenizer.add_tokens(new_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        if self.use_dual_encoder:
            self.encoder_location.resize_token_embeddings(len(self.tokenizer))

    def init_template(self, init_manual_template, original_template, num_tokens):
        if init_manual_template:
            template = self.convert_manual_to_dense(original_template)
        else:
            template = ' '.join(['[V%d]' % (i + 1) for i in range(num_tokens)]) + f' {CLASS_TOKEN} .'
        return template

    def convert_manual_to_dense(self, manual_template):
        def assign_embedding(new_token, token):
            """
            assign the embedding of token to new_token
            """
            logger.info('Tie embeddings of tokens: (%s, %s)' % (new_token, token))
            id_a = self.tokenizer.convert_tokens_to_ids([new_token])[0]
            id_b = self.tokenizer.convert_tokens_to_ids([token])[0]
            with torch.no_grad():
                self.encoder.embeddings.word_embeddings.weight[id_a] = self.encoder.embeddings.word_embeddings.weight[id_b].detach().clone()
                if self.use_dual_encoder:
                    self.encoder_location.embeddings.word_embeddings.weight[id_a] = self.encoder_location.embeddings.word_embeddings.weight[id_b].detach().clone()

        new_token_id = 0
        template = []
        for word in manual_template.split():
            if word == CLASS_TOKEN:
                template.append(word)
            else:
                tokens = self.tokenizer.tokenize(' ' + word)
                for token in tokens:
                    new_token_id += 1
                    template.append(get_new_token(new_token_id))
                    assign_embedding(get_new_token(new_token_id), token)

        return ' '.join(template)

    def load_model(self, model_name, config):
        if 'simcse' in model_name:
            encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif 'gpt' in model_name:
            encoder = GPT2Model(config)
        elif 'distilbert' in model_name:
            encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif 'spanbert' in model_name:
            encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif 'albert' in model_name:
            encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif 'bertweet' in model_name:
            encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False, config=config)
        elif 'longformer' in model_name:
            encoder = LongformerModel.from_pretrained(model_name, add_pooling_layer=False, config=config)
        elif 'roberta' in model_name:
            encoder = RobertaModel(config, add_pooling_layer=False)
        elif 'bert' in model_name:
            encoder = BertModel(config, add_pooling_layer=False)
        else:
            raise NotImplementedError(f'The given model name {model_name} is not implemented yet.')
        return encoder

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def get_tlc_loss(self, sim_score, labels):
        logits = sim_score.mean(-1)  # (batch_size, num_classes)
        if self.smooth_loss:
            loss_tlc = smooth_loss(logits, labels)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss_tlc = loss_fct(logits, labels)
        return loss_tlc, logits

    def log_sum_2_losses(self, loss_1, loss_2):
        return torch.log(torch.stack([loss_1, loss_2])).sum()

    def get_loss(self, sim_score, labels, feat_fused, location_pool):
        if self.use_tlc_loss and self.use_tlm_loss:
            loss_tlc, logits = self.get_tlc_loss(sim_score, labels)
            loss_tlm = self.loss_tlm(feat_fused, location_pool, labels, sim_score)
            wandb.log({'loss_tlc': loss_tlc})
            wandb.log({'loss_tlm': loss_tlm})
            loss = self.log_sum_2_losses(loss_tlc, loss_tlm) if self.log_loss else loss_tlc + loss_tlm

        elif self.use_tlc_loss and not self.use_tlm_loss:
            loss_tlc, logits = self.get_tlc_loss(sim_score, labels)
            wandb.log({'loss_tlc': loss_tlc})
            loss = loss_tlc

        elif not self.use_tlc_loss and self.use_tlm_loss:
            loss_tlm = self.loss_tlm(feat_fused, location_pool, labels, sim_score)
            wandb.log({'loss_tlm': loss_tlm})
            loss = loss_tlm
            logits = None

        else:
            raise ValueError('Use at least one loss')

        return loss, logits

    def forward(self, input_ids, attention_mask, cate_features, labels=None, inference=False):
        batch_size = input_ids.size(0)
        output_hidden_states = True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False

        feats_enc = []
        for idx_feat in range(input_ids.size(1)):
            cls = self.encoder(input_ids[:, idx_feat], attention_mask=attention_mask[:, idx_feat], output_hidden_states=output_hidden_states, return_dict=True)
            feats_enc.append(cls)

        tokenized_locations = {k: v.to(input_ids.device) for k, v in self.input_locations.items()}
        input_ids_labels = tokenized_locations['input_ids']  # (num_classes*num_templates, len), (330, 21)
        attention_mask_labels = tokenized_locations['attention_mask']
        location_enc = self.encoder(input_ids_labels, attention_mask=attention_mask_labels, output_hidden_states=output_hidden_states, return_dict=True) \
            if not self.use_dual_encoder else self.encoder_location(input_ids_labels, attention_mask=attention_mask_labels, output_hidden_states=output_hidden_states, return_dict=True)

        if not 'gpt' in self.model_name:
            feats_pool = [self.pooler(attention_mask[:, idx_feat, :], feat) for idx_feat, feat in enumerate(feats_enc)]
            location_pool = self.pooler(attention_mask_labels, location_enc)
        else:
            feats_pool = [self.pooler_gpt(feat, batch_size, input_ids[idx_feat], self.config.pad_token_id) for idx_feat, feat in enumerate(feats_enc)]
            location_pool = self.pooler_gpt(location_enc, input_ids_labels.size(0), input_ids_labels, self.config.pad_token_id)
        if self.pooler_type=='cls':
            feats_pool = [self.mlp(feat) for feat in feats_pool]
            location_pool = self.mlp(location_pool) if not self.use_dual_encoder else self.mlp_location(location_pool)
        if self.transformer_after_pooler:
            feats_pool = [self.transformer(feat) for feat in feats_pool]
            location_pool = self.transformer(location_pool) if not self.use_dual_encoder else self.transformer_location(location_pool)

        if cate_features.size(1)>0:
            cate_embeds = [self.cate_embedding0(cate_features[:, 0]), self.cate_embedding1(cate_features[:, 1])]
            feats_all = feats_pool+cate_embeds
        else:
            feats_all = feats_pool

        feat_fused = self.user_encoder(feats_all) if len(feats_pool) > 1 else feats_pool[0]
        # feat_fused = self.user_encoder(feats_all)
        sim_score = self.sim(feat_fused.unsqueeze(1), location_pool.unsqueeze(0))
        sim_score = sim_score.view(batch_size, self.num_classes, -1)

        if inference:
            logits = sim_score.mean(-1)
            return logits
        else:
            loss, logits = self.get_loss(sim_score, labels, feat_fused, location_pool)
            return SequenceClassifierOutput(loss=loss, logits=logits)