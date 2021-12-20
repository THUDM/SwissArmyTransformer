# roberta-base for SwissArmyTransformer

## Model Transformation

We transform [roberta-base](https://huggingface.co/roberta-base) in huggingface by `transform_param.py`.

```bash
python -m torch.distributed.launch transform_param.py
```