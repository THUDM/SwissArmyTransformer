## Introduction
`SwissArmyTransformer` is a flexible and powerful library to develop your own Transformer variants.

`SwissArmyTransformer` is named after "swiss army knife", meaning that all the models (e.g. BERT, GPT, T5, GLM, CogView, ViT...) **share the same backone code** and cater for versatile usages with some extra light-weight mixins. 

`SwissArmyTransformer` is powered by `deepspeed-ZeRO` and model parallelism, aiming to provide the best practice for pretraining and finetuning large models (100M\~20B parameters). 
## Install
```
    pip install SwissArmyTransformer
```
## Features

* Add model-agnostic components, e.g. prefix-tuning, in just *ONE* line! 

    - [Prefix-tuning](https://arxiv.org/pdf/2101.00190) (or [P-tuning](https://arxiv.org/abs/2103.10385)) improves finetuning via adding trainable parameters in each attention layer. To apply it to a [GLM](https://arxiv.org/pdf/2103.10360.pdf) classification (or any other) model is easy with our library.

    ```python
        class ClassificationModel(GLMModel):
            def __init__(self, args, transformer=None, **kwargs):
                super().__init__(args, transformer=transformer, **kwargs)
                self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
                # Arm an arbitrary model with Prefix-tuning with this line!
                self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    ```

    - GPT and other auto-regressive models act differently during training and inference. During inference, text is generated token-by-token and we need to cache previous states for efficiency. With our lib, you only need to consider the behavior during training (teacher-forcing) and transform it to a cached auto-regressive model via adding a mixin:

    ```python
        model = GLMModel(args)
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        # Generate a sequence with beam search
        from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
        from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy
        output, *mems = filling_sequence(model, input_seq,
                        batch_size=args.batch_size,
                        strategy=BeamSearchStrategy(args.batch_size))
    ```     


* Build your Transformer-based model with minimal codes. We mentioned [GLM](https://arxiv.org/pdf/2103.10360.pdf), which only differs from standard transformer (called BaseModel) on position embedding (and training losses). We only need to focus on the related part when coding.

    <details><summary>Extend the whole definition: </summary><p>

    ```python
    class BlockPositionEmbeddingMixin(BaseMixin):
        # Here define parameters for the mixin
        def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
            super(BlockPositionEmbeddingMixin, self).__init__()
            self.max_sequence_length = max_sequence_length
            self.hidden_size = hidden_size
            self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
        
        # Here define the method for the mixin
        def position_embedding_forward(self, position_ids, **kwargs):
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.transformer.position_embeddings(position_ids)
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            return position_embeddings + block_position_embeddings

    class GLMModel(BaseModel):
        def __init__(self, args, transformer=None, parallel_output=True):
            super().__init__(args, transformer=transformer, parallel_output=parallel_output)
            self.add_mixin('block_position_embedding', 
                BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size)
            ) # Add the mixin for GLM

        # we can also directly define hook-functions in the model.
        # E.g., The code below will remove position embeddings:

        # def position_embedding_forward(self, position_ids, **kwargs):
        #   return 0 

    ```

*  Comprehensive supports for training. `SwissArmyTransformer` aims to provide the best practice for pretraining and finetuning, where you only need to finish `forward_step` and `create_dataset_function` but with hyperparameters to alter useful training configurations.
    - Extend the training to multiple GPUs or nodes by specifying `--num_nodes`, `--num_gpus` and a simple `hostfile`. 
    - DeepSpeed and Model parallelism.
    - Better integration of ZeRO-2 and activation checkpointing.
    - Automatic extending and shuffling training data and `memmap`. 
    - Successfully support the training of [CogView2](http://github.com/THUDM/CogView).
    - The only open-source codebase supporting finetuning [T5-10B](https://arxiv.org/abs/1910.10683) on GPUs currently.

</p></details>


## Get started
```
    cd examples/cogview2
    ./scripts/text2image_cogview2.sh
```

### Run GLM
1. Prepare input.txt. Example: "Welcome! This is the main page of SwissArmyTransformer".
2. Run the following commands:
```
    cd examples/glm
    ./scripts/generate_glm.sh config/model_glm_10B_chinese.sh
```

Output:
[CLS]Welcome! This is the main page of SwissArmyTransformer. It is a comprehensive and clear explanation of the technical problems in the transformer. It is also an introduction to the development of the SwissArmy transformers. Welcome to Swiss Army Transforters. This is the main page of Swiss army tranforter. It's a complete and clean explaination of technology problem in the Tranformer, which is an integral part of the army's technological development. It also anintroduction of the developments of the Army technicians. Well, if you have any questions, please feel free to contact the official webs
