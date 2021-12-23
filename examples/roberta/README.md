# swiss-roberta

We transform parameters of [roberta-base](https://huggingface.co/roberta-base) and [roberta-large](https://huggingface.co/roberta-large) from huggingface to swiss by `transform_param.py`.

## Inference

```bash
bash scripts/inference_roberta.sh base
bash scripts/inference_roberta.sh large
```

## Finetune

```bash
bash scripts/finetune_boolq.sh base
bash scripts/finetune_boolq.sh large
```

### Performace

* roberta-base: iteration 1000 validation acc 0.7769
* roberta-large: iteration 1000 validation acc 0.8504
