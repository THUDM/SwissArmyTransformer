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


