# swiss-bert

We transform parameters of bert-base-uncased from huggingface to swiss by `transform_param.py`.

## Inference

```bash
bash scripts/inference_bert.sh
```

## Finetune

Finetune bert-base-uncased:

```bash
bash scripts/finetune_boolq.sh
```

* iter 1000 val acc: 0.7053

Finetune bert with adapter:

```bash
bash scripts/finetune_adapter_boolq.sh
```

* iter 1000 val acc: 0.6216