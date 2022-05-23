# CLIP

## Pretrained weights

You can download pretrained weights from [huggingface:openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

Then use `transform_param.py` to transform weights into `SwissArmyTransformer`.

```bash
python transform_param.py
```

## Inference

`inference_clip.ipynb`

or

```bash
bash scripts/inference_clip.sh /data/qingsong/pretrain
```

## Fine-tune

Here is an example of fine-tuning image encoder for CIFAR-10.

```bash
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iteration 1000 validation acc: 0.9456