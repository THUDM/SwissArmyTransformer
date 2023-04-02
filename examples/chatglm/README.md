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
python inference_adgen.py --mode inference --skip-init --fp16
```

Examples of finetuned model:

```
输入：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
输出：破洞牛仔裤是时尚达人的必备单品,这款牛仔裤在设计上,将破洞元素与刺绣元素结合在一起,简约的牛仔外套,打破原本的单调,更加符合年轻人的个性,同时,破洞的设计,更是彰显时尚个性。

输入：类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带裙*裙长#连衣裙*裙领型#半高领
输出：纯色的针织连衣裙,简洁大气的纯色设计,上身很显气质,复古文艺,穿上它秒变小仙女。简约的半高领设计,搭配小翻领,更显颈部曲线美,修饰脸型。背带的设计,加上后开叉的设计,穿上很显气质,时髦的

输入：类型#裙*裙下摆#弧形*裙腰型#高腰*裙长#半身裙*裙款式#不规则*裙款式#收腰
输出：这条半身裙的设计独特,弧形的腰部设计,凸显腰身曲线。收腰的腰身设计,勾勒出优美的腰身曲线,凸显女性身材。弧形的下摆设计,修饰出女性优美的腿部曲线。
```