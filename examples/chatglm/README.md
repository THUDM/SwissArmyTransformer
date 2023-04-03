# ChatGLM

This folder provides inference for [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model in SAT.

```
python transform_param.py
python inference_chatglm.py --mode inference --fp16 --device 1 --skip-init
python chat.py --mode inference --fp16 --skip-init
```

For finetune ChatGLM-6B, we adapt the [official finetune code](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning) to SAT:

```
bash scripts/finetune_adgen.sh
python inference_adgen.py --mode inference --skip-init --fp16 --ckpt_path checkpoints/finetune-chatglm-6b-adgen-04-03-06-48/
```

Examples of finetuned model:

```
输入：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
输出：经典款式的牛仔外套,简约中带有点点的设计感。采用破洞的设计,让整体看起来更加的帅气。白色刺绣的点缀,让整体看起来更加的精致。袖口和领口的破洞设计,让整体看起来更加的时髦。

输入：类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带裙*裙长#连衣裙*裙领型#半高领
输出：针织连衣裙是夏季必不可少的单品,纯色的针织面料,手感柔软细腻,穿着舒适,加上半高领的设计,穿着起来既保暖又舒适。搭配背带裙的设计,复古文艺,简约大方。

输入：类型#裙*裙下摆#弧形*裙腰型#高腰*裙长#半身裙*裙款式#不规则*裙款式#收腰
输出：高腰版型设计,将女性的身姿比例拉伸的更好,凸显女性修长的身材曲线,让女性更加有女人味。弧形下摆设计,凸显女性的优美气质,让女性更加优雅迷人。不规则下摆设计,修饰女性纤细的腿部,让女性更加有气场。
```