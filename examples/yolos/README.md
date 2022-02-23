# YOLOS

**YOLOS is adapted from https://github.com/hustvl/YOLOS**

## Pretrained weights

You can download pretrained weights from YOLOS repo, and then transform it into `SwissArmyTransformer` using `transform_param.py`.

```bash
python transform_param.py
```

## Inference

We adapt [VisualizeAttention.ipynb](https://github.com/hustvl/YOLOS/blob/main/VisualizeAttention.ipynb) here as `inference_yolos.ipynb`.

## Training

```bash
bash scripts/finetune_coco.sh
```

For now, we haven't support customized evaluator. Therefore, I just report loss as metric.