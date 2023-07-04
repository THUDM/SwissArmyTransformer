# ChatGLM2

This folder provides [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) model in SAT.

If you want to chat with ChatGLM2-6B with huggingface generation:

```
python chat.py --mode inference --fp16 --skip-init
```

Models will be downloaded and cached automatically into `~/.sat_models`.

Here is the parameter transformation script (from huggingface/chatglm2-6b to sat/chatglm2-6b):

```
python transform_param.py
```