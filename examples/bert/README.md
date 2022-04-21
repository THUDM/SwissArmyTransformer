# swiss-bert

We transform parameters of bert-base-uncased from huggingface to swiss by `transform_param.py`.

## Inference

```bash
bash scripts/inference_bert.sh
```

## Finetune

```bash
bash scripts/finetune_boolq.sh
```

* iter 1000 val acc: 0.7053