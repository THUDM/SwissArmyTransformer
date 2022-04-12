# CLIP

## Pretrained weights

You can download pretrained weights from [huggingface:openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

Then use `transform_param.py` to transform weights into `SwissArmyTransformer`.

```bash
python transform_param.py
```

## Inference

`inference_clip.ipynb`