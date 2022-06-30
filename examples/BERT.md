# BERT

Welcome use `examples/bert` as a reference to contribute new models to `SwissArmyTransformer`.

[Here](https://github.com/THUDM/SwissArmyTransformer/tree/main/SwissArmyTransformer/model/official) are some officially implemented models including BERT.

You can build your own model by adding and deleting mixins to some base model like [this one](https://github.com/THUDM/SwissArmyTransformer/blob/main/examples/bert/bert_ft_model.py). Then use your model to train on some dataset like [this script](https://github.com/THUDM/SwissArmyTransformer/blob/main/examples/bert/finetune_bert_boolq.py) follwing [training tutorial](/tutorials/training.md).

There are also some example of training with [adapter](https://github.com/THUDM/SwissArmyTransformer/blob/main/examples/bert/finetune_bert_adapter_boolq.py) or [distillation](https://github.com/THUDM/SwissArmyTransformer/blob/main/examples/bert/finetune_distill_boolq.py).