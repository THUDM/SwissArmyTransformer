import os
import sys
import torch

def megatron_to_sat(checkpoint_name, target):
    num_layer = 70
    # checkpoint_name = '/thudm/workspace/hanyu/SwissArmyTransformer-old/data/global_step10400/iter_0010400/10400/mp_rank_00_model_states.pt'
    sd = torch.load(checkpoint_name, map_location='cpu')
    new_sd = {}
    new_sd['transformer.word_embeddings.weight'] = sd['model']['word_embeddings_for_head']['weight']

    encoder = sd['model']['language_model']['encoder']

    for i in range(num_layer):
        new_sd['transformer.layers.' + str(i) +'.input_layernorm.weight'] = encoder['layers.' + str(i) + '.input_layernorm.weight']
        new_sd['transformer.layers.' + str(i) +'.input_layernorm.bias'] = encoder['layers.' + str(i) + '.input_layernorm.bias']

        new_sd['transformer.layers.' + str(i) + '.attention.query_key_value.weight'] = encoder['layers.' + str(i) + '.self_attention.query_key_value.weight']
        new_sd['transformer.layers.' + str(i) + '.attention.query_key_value.bias'] = encoder['layers.' + str(i) + '.self_attention.query_key_value.bias']

        new_sd['transformer.layers.' + str(i) + '.attention.dense.weight'] = encoder['layers.' + str(i) + '.self_attention.dense.weight']
        new_sd['transformer.layers.' + str(i) + '.attention.dense.bias'] = encoder['layers.' + str(i) + '.self_attention.dense.bias']

        new_sd['transformer.layers.' + str(i) + '.post_attention_layernorm.weight'] = encoder['layers.' + str(i) + '.post_attention_layernorm.weight']
        new_sd['transformer.layers.' + str(i) + '.post_attention_layernorm.bias'] = encoder['layers.' + str(i) + '.post_attention_layernorm.bias']

        new_sd['transformer.layers.' + str(i) + '.mlp.dense_h_to_4h.weight'] = encoder['layers.' + str(i) + '.mlp.dense_h_to_4h.weight']
        new_sd['transformer.layers.' + str(i) + '.mlp.dense_h_to_4h.bias'] =  encoder['layers.' + str(i) + '.mlp.dense_h_to_4h.bias']

        new_sd['transformer.layers.' + str(i) + '.mlp.dense_4h_to_h.weight'] =  encoder['layers.' + str(i) + '.mlp.dense_4h_to_h.weight']
        new_sd['transformer.layers.' + str(i) + '.mlp.dense_4h_to_h.bias'] =  encoder['layers.' + str(i) + '.mlp.dense_4h_to_h.bias']

    new_sd['transformer.final_layernorm.weight'] = encoder['final_layernorm.weight']
    new_sd['transformer.final_layernorm.bias'] = encoder['final_layernorm.bias']
    new_sd = { 'module': new_sd }
    # target = open('/thudm/workspace/hanyu/SwissArmyTransformer/data/global_step10400/iter_0010400/10400/mp_rank_00_model_states.pt', 'w')
    torch.save(new_sd, target)


def main():
    dir_path = str(sys.argv[1])
    target_dir = str(sys.argv[2])

    iter_path = os.path.join(dir_path, 'latest_checkpointed_iteration.txt')

    iteration = open(iter_path).read()

    print(iteration)

    new_iter_dir = os.path.join(target_dir, 'iter_00' + iteration)
    iter_dir = os.path.join(dir_path, 'iter_00' + iteration)

    os.mkdir(new_iter_dir)
    new_iter_path = os.path.join(new_iter_dir, 'latest')

    print(iter_path, new_iter_path)
    os.system('cp {} {}'.format(iter_path, new_iter_path))

    new_model_dir = os.path.join(new_iter_dir, iteration)
    os.mkdir(new_model_dir)

    for i in range(8):
        model_dir = os.path.join(iter_dir, 'mp_rank_0' + str(i))
        model_path = os.path.join(model_dir, 'model_optim_rng.pt')
        new_model_path = os.path.join(new_model_dir, 'mp_rank_0' + str(i) + '_model_states.pt')
        print(model_path, new_model_path)
        megatron_to_sat(model_path, new_model_path)
    
if __name__ == "__main__":
    main()