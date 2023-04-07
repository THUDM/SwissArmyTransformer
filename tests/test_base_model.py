from SwissArmyTransformer.model.base_model import BaseModel
if __name__ == '__main__':
    from argparse import Namespace
    from SwissArmyTransformer.arguments import _simple_init
    args = Namespace(
        num_layers=2,
        vocab_size=100,
        hidden_size=100,
        num_attention_heads=2,
        max_sequence_length=100,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        output_dropout_prob=0.1,
        inner_hidden_size=100,
        hidden_size_per_attention_head=50,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        layernorm_order='pre',
        skip_init=False,
        use_gpu_initialization=False,
        model_parallel_size=1,
    )
    _simple_init(args.model_parallel_size)

    m1 = BaseModel(args)