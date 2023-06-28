# LLaMA

You can run inference for llama model by:

```
python inference.py --mode inference --skip-init --fp16
python chat_sat.py
```

Here is the weight transformation script from [huggingface weight](https://huggingface.co/docs/transformers/main/model_doc/llama):

```
python transform_param.py
```