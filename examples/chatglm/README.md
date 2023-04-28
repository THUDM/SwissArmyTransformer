# ChatGLM

This folder provides inference and fineuning for [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model in SAT.

If you want to inference or chat with ChatGLM-6B:

```
python chat_sat.py
```

If you want to inference or chat with ChatGLM-6B with huggingface generation instead of sat (legacy):
```
python inference_chatglm.py --mode inference --fp16 --device 0 --skip-init
python chat.py --mode inference --fp16 --skip-init
```

Models will be downloaded and cached automatically into `~/.sat_models`.

For finetuning 0 ChatGLM-6B, we adapt the [official finetune code](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning) to SAT:

```
bash scripts/finetune_adgen.sh
python inference_adgen.py --mode inference --skip-init --fp16 --ckpt_path checkpoints/finetune-chatglm-6b-adgen-04-03-06-48/
```

Examples of finetuned model:

```
输入：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
输出：破洞牛仔裤是时尚界永不过时的单品,在牛仔裤上加入刺绣图案,既增添层次感,又增添了一份时尚感,上身后尽显俏皮俏皮的青春活力。白色的面料,干净简约,穿着舒适。

输入：类型#裙*材质#针织*颜色#纯色*风格#复古*风格#文艺*风格#简约*图案#格子*图案#纯色*图案#复古*裙型#背带裙*裙长#连衣裙*裙领型#半高领
输出：纯色的连衣裙,简约大方,没有过多的装饰,看起来十分低调却不失时尚。半高领的设计,可以拉长脖子,修饰脸型,显得脸小。背带的设计,可以凸显女性纤细的身材,凸显气质。格子的样式,复古文艺,充满了浓浓的复古气息,让你看起来与众不同

输入：类型#裙*裙下摆#弧形*裙腰型#高腰*裙长#半身裙*裙款式#不规则*裙款式#收腰
输出：非常时髦的半身裙,弧形的高腰设计,修饰腰身,凸显身材曲线美,同时收腰的设计,修饰腰部,尽显女性的优美曲线美。不规则裙摆的设计,增添了几分灵动感和时尚感,让整件裙子更加有层次感,又带着几分设计感。
```

Here is the parameter transformation script (from huggingface/chatglm-6b to sat/chatglm-6b):

```
python transform_param.py
```