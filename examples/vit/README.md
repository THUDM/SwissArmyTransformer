# swiss-vit

We transform pretrained ViT parameters from [timm](https://github.com/rwightman/pytorch-image-models) using `transform_param.py`.

## Finetune

```bash
bash scripts/finetune_cifar10.sh
```

* vit-base-224-16-21k: iteration 2000 validation acc 0.9647