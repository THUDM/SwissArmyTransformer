import os
os.environ["TRANSFORMERS_OFFLINE"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

pretrain_path = '/workspace/bert'


from transformers import BertTokenizer, BertModel, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrain_path, 'bert-base-uncased'))
bert = BertForSequenceClassification.from_pretrained(os.path.join(pretrain_path, 'bert-base-uncased'), output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", do_lower_case=True)

# lm_head = roberta.lm_head
# roberta = roberta.roberta
def copy_layer_param(src, dst):
    """
    in-place copy from src to dst
    src and dst should be the same layer type, e.g., both are LayerNorm or both are Linear.
        Or at least, both have same named_parameters name and shape.
    """
    src_dic = dict(src.named_parameters())
    dst_dic = dict(dst.named_parameters())
    for k in dst_dic:
        assert dst_dic[k].data.shape == src_dic[k].data.shape
        dst_dic[k].data = src_dic[k].data
        assert (dst_dic[k].data == src_dic[k].data).all()

def copy_layer_norm(src, dst):
    src_ln = []
    for k, v in src.named_parameters():
        if 'layernorm' in k.lower():
            src_ln.append((k, v))
    dst_ln = []
    for k, v in dst.named_parameters():
        if 'layernorm' in k.lower():
            dst_ln.append((k, v))
    assert len(src_ln) == len(dst_ln)
    for kvs, kvd in zip(src_ln, dst_ln):
        assert kvd[1].data.shape == kvs[1].data.shape
        kvd[1].data = kvs[1].data
        assert (kvd[1].data == kvs[1].data).all()

def copy_transformer_layer_wo_ln(src, dst):
    new_weight = torch.cat([src.attention.self.query.weight.data, src.attention.self.key.weight.data, src.attention.self.value.weight.data], 0)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = torch.cat([src.attention.self.query.bias.data, src.attention.self.key.bias.data, src.attention.self.value.bias.data], 0)
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attention.output.dense, dst.attention.dense)
    copy_layer_param(src.intermediate.dense, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.output.dense, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.embeddings.word_embeddings, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.embeddings.position_embeddings, swiss_model.transformer.position_embeddings)
    copy_layer_param(hugging_model.embeddings.token_type_embeddings,
                     model.mixins['position_embedding_forward'].token_type_embeddings)
    # swiss_model.transformer.word_embeddings.padding_idx = roberta.embeddings.padding_idx
    # swiss_model.transformer.position_embeddings.padding_idx = roberta.embeddings.padding_idx
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l in zip(hugging_model.encoder.layer, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(hugging_model.pooler.dense, model.mixins['final_forward'].lm_head.dense1)
    copy_layer_param(lm_head, model.mixins['final_forward'].lm_head.dense2)

if __name__ == "__main__":
    lm_head = bert.classifier
    bert = bert.bert
    import argparse
    args = argparse.Namespace(
        num_layers=12,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        max_sequence_length=512,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        sandwich_ln=False,
        model_parallel_size=1,
        world_size=1,
        rank=0,
        output_size=2,
        use_final_layernorm = False
    )

    import torch
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '16677')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank, init_method=init_method)

    import SwissArmyTransformer.mpu as mpu
    mpu.initialize_model_parallel(args.model_parallel_size)

    from bert_model import BertModel

    model = BertModel(args)


    from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

    bert.eval()
    model.eval()
    with torch.no_grad():
        transform_weight(bert, model)
        text = ["This is a piece of text.", "Another piece of text."]
        encoded_input = tokenizer(text, return_tensors='pt', padding=True)
        #position_ids = create_position_ids_from_input_ids(encoded_input['input_ids'], bert.embeddings.padding_idx, 0)
        #print(position_ids)
        position_ids=torch.arange(len(encoded_input['input_ids'][0])).expand((len(encoded_input)-1, -1))
        breakpoint()
        output = bert(**encoded_input)
        hugging_output = lm_head(output.pooler_output)
        model.cuda()
        attention_mask = encoded_input['attention_mask'][:, None, None, :]
        swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
        print("max error:", (hugging_output[0] - swiss_output[0]).abs().max())
        print("max relative error:", ((hugging_output[0] - swiss_output[0]).abs() / torch.max(swiss_output[0].abs(), hugging_output[0].abs())).max())
        torch.save(model.state_dict(), os.path.join(pretrain_path, "bert-base.pt"))

    # breakpoint()