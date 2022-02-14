# DeiT

DeiT shares the same architecture with ViTModel. Therefore, you can use `transform_param.py` to transform pretrained DeiT into `SwissArmyTransformer`, and then use it in a same way as in [examples/vit](../vit).

For now, we support three deit types, i.e., deit-tiny, deit-small and deit-base.

```bash
python transform_param.py --model tiny
python transform_param.py --model small
python transform_param.py --model base
```

For more model types, you may refer to [deit in timm](https://github.com/facebookresearch/deit/blob/main/models.py#L63).