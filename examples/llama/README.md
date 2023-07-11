# LLaMA

You can run inference for llama model by:

```
python inference.py --mode inference --skip-init --fp16
python chat_sat.py
```

For 30b and 65b, you need to use model parallel to inference:

```
torchrun --standalone --nnodes=1 --nproc-per-node=8 split_model.py
```

Here is the weight transformation script from [huggingface weight](https://huggingface.co/docs/transformers/main/model_doc/llama):

```
python transform_param.py
```