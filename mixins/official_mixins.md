# Official Mixins

## MLP Head

```python
class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005):
```

For example, MLP head with input layer 256, two hidden_layers 768, and output layer 256:

```python
from SwissArmyTransformer.model.mixins import MLPHeadMixin
MLPHeadMixin(256, 768, 768, 256)
```

To add the mixin into your model:

```python
class ExampleModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        self.del_mixin('old_head_name')
        self.add_mixin('new_head', MLPHeadMixin(args.hidden_size, 768, 768, 256))
```

## Prompt Tuning

```python
class PrefixTuningMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size_per_attention_head, num_attention_heads, prefix_len):
```

Example:

```python
from SwissArmyTransformer.model.mixins import PrefixTuningMixin 
class ExampleModel(BertModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        self.del_mixin('old_attn_fn_mixin')
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('example-finetune', 'example finetune Configurations')
        group.add_argument('--prefix_len', type=int, default=16)
```