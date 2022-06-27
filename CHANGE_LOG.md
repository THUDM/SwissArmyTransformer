# 2021.10.29 v0.1
1. change `mixins` from `ModuleList` to `ModuleDict`
2. return tokens and mems in `fill_sequence`, and mems becomes a tensor.
3. `CachedAutoRegressiveMixin`
## How to migrate old SAT ckpt to new version?
Example:
```python
import torch
old = torch.load('xxxxx/mp_rank_00_model_states.pt.old', map_location='cpu')

# replace names, mixins index to keys
oldm = old['module']
for k in list(oldm.keys()):
    if k.startswith('mixins.0'):
        new_k = k.replace('mixins.0', 'mixins.extra_position_embedding')
    elif k.startswith('mixins.1'):
        new_k = k.replace('mixins.1', 'mixins.attention_plus')
    else:
        continue
    oldm[new_k] = oldm[k]
    del oldm[k]
# save to destination    
torch.save(old, 'xxxxx/mp_rank_00_model_states.pt')

```
for the older framework, you also need:
```python
old['module']['transformer.word_embeddings.weight'] = old['module']['word_embeddings.weight']
del old['module']['word_embeddings.weight']
```
# 2021.11.5 v0.1.2
1. Add generation.autoregressive_sampling.evalute_perplexity
2. fix Runtime Error in skipping Nan Loss

# 2021.12.13 v0.1.4
1. Add non_conflict attention_fn
2. Add Prefix-Tuning
3. Now, you can use `kw_args['output_this_layer']` (any hooks in the transformer layers) to return values to final outputs and `kw_args['output_cross_layer']` to pass values to `kw_args` in the next layer.

Examples:
```
def attention_fn(...some_args):
    ...
    kw_args['output_this_layer']['mem_kv'] = cache_kv
    ...
```
This will let the key `'mem_kv'` appear in the `outputs_per_layers[i]` of `logits, *outputs_per_layers = model(...)`. 

```
def attention_fn(...some_args, **kw_args):
    ...
    kw_args['output_cross_layer']['last_attention_map'] = attention_map
    ...
```
This will let the key `'last_attention_map'` appear in the next layer's `kw_args` (all hooks). 

# 2021.12.13 v0.1.7
1. Ensure enough training data, no longer always 200 times
2. You can use `kw_args['cross_layer_output']['new_key']=xxx` to pass other results to each layer in `position/word_embedding_forward`.
3. Add `--train-data-weights`.

# 2022.1.13 v0.1.9
1. Add Vit
2. Fix evaluation all_reduce bug

# 2022.6.3 v2.0
1. split all the default hooks out
2. change the order, model hooks will not override all the things. They now are the same as mixin hooks added in the **front** of all the mixins.

# 2022.6.6 v2.0
1. `from_pretrained` now auto downloads models. There are two kinds of usages: `SomeModel.from_pretrained(args, name)` will load the weights of `name` model to a `SomeModel` with the same model arch hyper-params with `name`; `AutoModel.from_pretrained(args, name)` will return an official model (`model_class` Class) with the pretrained weights.
2. ENV `SAT_HOME` is where we put the models in. Set it in your shell file.
3. don't necessarily need `deepspeed_config`, or pass model arch hyper-params for `from_pretrained`. Use `zero-stage 0/1/2`.  

# 2022.6.27
1. Fix *flat_output bug.
2. fix defualt mpu init_method bug.