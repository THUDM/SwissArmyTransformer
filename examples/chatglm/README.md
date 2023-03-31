# ChatGLM

This folder provides inference for [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model in SAT.

```
python transform_param.py
python inference_chatglm.py --mode inference --fp16 --device 1 --skip-init
python chat.py --mode inference --fp16 --skip-init
```