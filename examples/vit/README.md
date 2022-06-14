# swiss-vit

We transform pretrained ViT parameters from [timm](https://github.com/rwightman/pytorch-image-models) using `transform_param.py`.

## Finetune

### CIFAR-10

```bash
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* vit-base-224-16-21k: iteration 1000 validation acc
     * online mode: 0.9854
     * offline mode: 0.9862

### ImageNet-1k

```bash
bash scripts/finetune_imagenet.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* vit-base-224-16-21k: iteration 1000 validation acc: 0.7995