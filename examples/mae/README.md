# MAE

## Pretrained weights

You can download pretrained weights from [facebookresearch/mae](https://github.com/facebookresearch/mae/issues/8).

Then use `transform_param.py` to transform weights into `SwissArmyTransformer`.

```bash
python transform_param.py
```

## Inference

`inference_mae.ipynb`

## Fine-tune Encoder

```bash
bash scripts/finetune_cifar10.sh
```

* mae-base: iteration 1000:
    * acc 0.5703