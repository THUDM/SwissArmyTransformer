# swiss-vit

We transform pretrained ViT parameters from [timm](https://github.com/rwightman/pytorch-image-models) using `transform_param.py`.

## Finetune

```bash
bash scripts/finetune_cifar10.sh
```

* vit-base-224-16-21k: iteration 1000 validation acc
    * online mode: 0.9880
    * offline mode: 0.9904
