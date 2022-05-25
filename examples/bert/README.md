# swiss-bert

We transform parameters of bert-base-uncased from huggingface to swiss by `transform_param.py`.

**Replace /data/qingsong/pretrain to your model root and /data/qingsong/dataset to your data root!**

## Inference

```bash
bash scripts/inference_bert.sh /data/qingsong/pretrain
```

## Finetune

Finetune bert-base-uncased:

```bash
bash scripts/finetune_boolq.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iter 1000 val acc: 0.7138

Finetune bert with adapter:

```bash
bash scripts/finetune_adapter_boolq.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iter 1000 val acc: 0.6216