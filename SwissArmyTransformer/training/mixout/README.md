# Implementation of Mixout with PyTorch
This repository contains a PyTorch code of mixout. This technique regularizes learning to minimize the deviation from the target parameters. For more detailed description of mixout, see ["Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"](https://arxiv.org/abs/1909.11299).       

![Mixout](imgs/mixout.png "Mixout")

# How to use

There is an example code (**example.py**) about applying mixout to a model. In **mixout.py**, you can find the functional version of mixout similar to *torch.nn.functional.dropout*. The module version of mixout is available in **module.py** as well, but it is quite different compared to *torch.nn.Dropout*. I highly recommend users to read **example.py**.

Thanks to Michael Wilson, there is also an example of applying mixout to a pretrained model from Huggingface in **example_huggingface.py**. Because of how models on Huggingface are structured, this works slightly differently from **example.py**.

# Reference
Cheolhyoung Lee, Kyunghyun Cho, and Wanmo Kang, Mixout: Effective regularization to Finetune Large-scale Pretrained Language Models, _International Conference on Learning Representations_ (2020).

# Additional Information
Stephen Roller also implemented mixout [in his gist](https://gist.github.com/stephenroller/f45a372e231825f9f5578e9e705f4e95). His implementation is actually mixconnect similar to dropconnect. (It is also introduced in the mixout paper.) However, unlike my implementation, *MixWrapper* can wrap most of *torch.nn.Module*'s and that you do not need to make your mixed module such as *MixLinear* in **mixlinear.py**. If you do not need to customize mixout, his code is convenient to use.       
