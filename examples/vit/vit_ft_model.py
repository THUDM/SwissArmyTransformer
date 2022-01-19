import torch
from SwissArmyTransformer.model.base_model import BaseMixin, non_conflict
from SwissArmyTransformer.model import ViTModel
from SwissArmyTransformer.model.mixins import PrefixTuningMixin

class NewClassHeadMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.classifier = torch.nn.Linear(args.hidden_size, args.num_finetune_classes)

class ViTFinetuneModel(ViTModel):
    def __init__(self, args, transformer=None, parallel_output=True, layernorm_epsilon=1e-6):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('finetune_head', NewClassHeadMixin(args))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    def final_forward(self, logits, **kw_args):
        logits = self.mixins["finetune_head"].classifier(logits[:, 0])
        return logits

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT-finetune', 'ViT finetuning Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        return super().add_model_specific_args(parser)