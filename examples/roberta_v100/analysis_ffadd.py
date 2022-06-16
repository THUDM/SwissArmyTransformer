# -*- encoding: utf-8 -*-
# @File    :   analyis_head
# @Time    :   2022/4/6
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import numpy
import torch
import argparse
import numpy as np
import copy
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main, initialize_distributed, load_checkpoint
from SwissArmyTransformer.model.finetune import *
from SwissArmyTransformer.model.mixins import BaseMixin
from functools import partial
from utils import create_dataset_function, ChildTuningAdamW, set_optimizer_mask
import os
from roberta_model import RobertaModel
# from utils import *
# from tqdm import tqdm

from transformers import RobertaTokenizer
pretrain_path = ''
tokenizer =  RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'), local_files_only=True)

class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu):
        super().__init__()
        # init_std = 0.1
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        logits = logits[:,1:2].sum(1)
        return logits
        #直接返回模型输出
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        # self.add_mixin('ffadd', FFADDMixin(args.hidden_size, args.num_layers, args.ffadd_r))
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))


from SwissArmyTransformer.data_utils import make_loaders
from SwissArmyTransformer.training.utils import Timers
from utils import create_dataset_function
def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'attention_mask', 'label']
    datatype = torch.int64
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['label'].long()
    position_ids = data_b['position_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, attention_mask, position_ids, (tokens!=1)

def solve_ffadd(args):
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-sst2-ffadd-lr0.0005-seed493092768-05-20-01-35"
    model = ClassificationModel(args)
    _ = load_checkpoint(model, args)
    model.requires_grad_(False)
    model.to('cuda:0')

    good_mixin = model.mixins["ffadd"]

    dataset_name = args.dataset_name

    args.train_data=None
    args.valid_data=[f"hf://glue/{dataset_name}/validation"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    # print("hahahahh")
    timers = Timers()

    data_num = len(val_data)
    val_data = iter(val_data)
    ffadd_r = 32
    sentences = []
    words = []
    positive = [[] for i in range(24)]
    thre = 2
    for i in range(24):
        for j in range(ffadd_r):
            positive[i].append([])
    # data_num = 800
    for i in tqdm(range(data_num)):
        tokens, labels, attention_mask, position_ids, loss_mask = get_batch(val_data, args, timers)
        # attention_output = []
        # output_good = model(tokens, position_ids, attention_mask, attention_output = attention_output)
        # for k in range(24):
        #     attention_output.append(output_good[k+1]["0"])
        # now_pos = len(sentences)
        # now_word_pos = len(words)
        for j in range(tokens.shape[0]):
            sentences.append(tokenizer.decode(tokens[j]))
            for k in range(len(tokens[j])):
                words.append(tokenizer.decode(tokens[j][k]))
        # for j in range(24):
        #     for k in range(attention_output[j].shape[1]):
        #         for l in range(ffadd_r):
        #             value = attention_output[j][0,k,l]
        #             if value > thre:
        #                 #pos is j,l, sentence is now_pos, k
        #                 positive[j][l].append((now_pos, now_word_pos+k))
    import json
    with open(f"sst2_thr{thre}.json", "r") as f:
        positive = json.load(f)

    # import json
    # with open(f"sst2_thr{thre}.json", "w") as f:
    #     json.dump(positive, f)

    lens = []
    ll = 4
    rr = 60
    answers = []
    for i in range(24):
        for j in range(32):
            if len(positive[i][j])>ll and len(positive[i][j])<rr:
                answers.append((i,j,positive[i][j]))

    outputs = []
    for id_i, id_j, array in answers:
        print("let!!!!!")
        concat = []
        for pos_s, pos_w in array:
            if words[pos_w] != "<pad>":
                concat.append(sentences[pos_s] + "               " + words[pos_w])
        outputs.append([id_i, id_j, concat])

    with open(f"sst2_thr{thre}_output.json", "w") as f:
        json.dump(outputs, f)

def solve_tsne_head(args):
    model_head = ClassificationModel(args)
    # args.load = '/thudm/workspace/yzy/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-ffadd-lr0.0005-seed944257842-05-16-14-17'
    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39' #rte
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-cb-2step+bitfit-lr1e-05-seed467940855-new2-05-23-21-39"
    _ = load_checkpoint(model_head, args) #头对但是bias不对
    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-pt[0-0]-lr0.007-seed765780494-05-16-17-55' #rte
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-cb-2step+pt-lr1e-05-seed516608503-new2-05-23-23-29"
    model_new = ClassificationModel(args)
    _ = load_checkpoint(model_new, args) #bias对但是头不对


    model_head.requires_grad_(False)
    model_new.requires_grad_(False)
    model_new.to('cuda:0')
    model_head.to('cuda:0')

    for i in range(2):
        old_weights = model_head.mixins["classification_head"].layers[i].weight.data
        model_new.mixins["classification_head"].layers[i].weight.data.copy_(old_weights)
        torch.nn.init.normal_(model_new.mixins["classification_head"].layers[i].weight, mean=0, std=0.005)

    args.train_data= None
    args.valid_data=[f"hf://super_glue/{args.dataset_name}/train"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    timers = Timers()
    data_num = len(val_data)
    val_data = iter(val_data)
    cls_list = []
    label_list = []
    for i in tqdm(range(data_num)):
        tokens, labels, attention_mask, position_ids, loss_mask = get_batch(val_data, args, timers)
        attention_output = []
        output_good = model_new(tokens, position_ids, attention_mask, attention_output = attention_output)[0].data.cpu().numpy()
        cls_list.append(output_good[0])
        label_list.append(labels[0].data.cpu().numpy())
    cls_list = numpy.stack(cls_list)
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    good_embedding = TSNE(n_components=2).fit_transform(cls_list)
    plt.scatter(good_embedding[:, 0], good_embedding[:, 1], s=20, c=label_list)
    plt.savefig(f'images/bad_head_train_{args.dataset_name}.jpg')
    plt.clf()

def solve_draw_diff(args):
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-pt[0-0]-lr0.007-seed784846687-05-16-18-05"
    model_old = ClassificationModel(args)
    _ = load_checkpoint(model_old, args) #bias对但是头不对
    model_new = ClassificationModel(args)
    # args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-all-lr1e-05-seed504467621-05-30-16-53"
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-all-lr1e-05-seed495704012-savemore-05-30-21-37"
    latest = args.load + '/latest'
    # with open(latest, "r") as f:
    #     iter = int(f.readline())//100
    iter = 46
    diff_list = []

    para_1 = {}
    for name, params in model_old.named_parameters():
        if 'transformer.layers' in name:
            para_1[name] = params.data
    for i in range(1, iter+1):
        #load model
        print("now is ", i)
        with open(latest, "w") as f:
            f.write(str(i*100))
        _ = load_checkpoint(model_new, args)
        #calc diff
        total_diff = 0
        for name, params in model_new.named_parameters():
            if 'transformer.layers' in name and name in para_1:
                diff = para_1[name] - params.data
                diff = (diff * diff).sum()
                total_diff += diff
        total_diff = math.sqrt(total_diff)
        diff_list.append(total_diff)
        print(total_diff)
    print(diff_list)

def draw_hist():
    para_list = np.load("para_list_random.npy")
    super_threshold_indices = para_list < 1e-7
    para_list[super_threshold_indices] = 1e-7
    super_threshold_indices = para_list > 1
    para_list[super_threshold_indices] = 1

    import matplotlib.pyplot as plt
    hist, bins = np.histogram(para_list, bins=30)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(para_list, bins=logbins, facecolor='g', alpha=0.75)
    plt.xscale('log')
    # plt.xlabel('梯度绝对值大小')
    # plt.ylabel('占比')
    # plt.title('模型初始状态下梯度绝对值分布')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.savefig("2step+bitfit-hist.jpg")
    breakpoint()

def calc_mask(model, args, train_data, forward_step):
    timers = Timers()
    # N = len(train_data)
    N = 78
    print(f"{N} samples to calc mask")

    model.train()
    gradient = dict()
    for name, params in model.named_parameters():
        if 'transformer.layers' in name:
            gradient[params] = params.new_zeros(params.size())
    for _ in tqdm(range(N)):
        loss, _ = forward_step(train_data, model, args, timers)
        loss.backward()

        for name, params in model.named_parameters():
            if 'transformer.layers' in name:
                # torch.nn.utils.clip_grad_norm_(params, 10)
                gradient[params] += params.grad
        model.zero_grad()

    para_list = []
    for name, params in model.named_parameters():
        if 'transformer.layers' in name:
            para_list.append(gradient[params].cpu().numpy().flatten())
            # total_sum += (gradient[params] ** 2).sum()
            # sz = gradient[params].size()
            # cnt2 = 1
            # for szz in sz:
            #     cnt2 *= szz
            # print(name[18:], f"{cnt}/{cnt2}", (cnt/cnt2).cpu().numpy().tolist())
    para_list = numpy.concatenate(para_list, axis=0)

    numpy.save("para_list_random", para_list)

    # draw_hist()
    # print(total_sum)

def forward_step(data_iterator, model, args, timers):
    tokens, labels, attention_mask, position_ids, loss_mask, *extra_data = get_batch(
        data_iterator, args, timers)
    # timers('batch generator').stop()
    if len(extra_data) >= 1:
        extra_data = extra_data[0]
    else:
        extra_data = {}
    attention_output = []
    logits, *mems = model(tokens, position_ids, attention_mask, attention_output = attention_output, **extra_data)
    pred = logits.contiguous().float().squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float(),
    )
    return loss, None

def calc_gradient(args):
    model_head = ClassificationModel(args)
    # args.load = '/thudm/workspace/yzy/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-ffadd-lr0.0005-seed944257842-05-16-14-17'
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39' #rte

    # args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-cb-2step+bitfit-lr1e-05-seed467940855-new2-05-23-21-39"
    _ = load_checkpoint(model_head, args) #头对但是bias不对
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-all-lr1e-05-seed495704012-savemore-05-30-21-37' #rte
    # args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-cb-2step+pt-lr1e-05-seed516608503-new2-05-23-23-29"



    model_new = ClassificationModel(args)
    _ = load_checkpoint(model_new, args) #bias对但是头不对

    model_new.to('cuda:0')
    model_head.to('cuda:0')

    # for i in range(2):
    #     old_weights = model_head.mixins["classification_head"].layers[i].weight.data
    #     model_new.mixins["classification_head"].layers[i].weight.data.copy_(old_weights)
        # torch.nn.init.normal_(model_new.mixins["classification_head"].layers[i].weight, mean=0, std=0.005)
    args.train_data= None
    args.valid_data=[f"hf://super_glue/{args.dataset_name}/train"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    val_data = iter(val_data)
    print("no random is !!!!")
    calc_mask(model_new, args, val_data, forward_step)

def calc_dir(args):
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-pt[0-0]-lr0.007-seed784846687-05-16-18-05"
    model_old = ClassificationModel(args)
    _ = load_checkpoint(model_old, args) #原参数

    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39' #rte
    model_new = ClassificationModel(args)
    _ = load_checkpoint(model_new, args) #bias对但是头不对

    para_1 = {}
    for name, params in model_old.named_parameters():
        if 'transformer.layers' in name:
            para_1[name] = params.data
    _ = load_checkpoint(model_new, args)
    #calc diff
    total_diff = 0
    para_list = []
    for name, params in model_new.named_parameters():
        if 'transformer.layers' in name and name in para_1:
            diff = para_1[name] - params.data
            para_list.append(diff.cpu().numpy().flatten())
    para_list = np.concatenate(para_list, axis=0)
    numpy.save("para_diff_2stepbit_rte", para_list) #这里存反了

def calc_dir2(args):
    p11 = numpy.load("para_list_random_2stepbit_rte.npy")
    p12 = numpy.load("para_diff_2stepbit_rte.npy")
    p21 = numpy.load("para_list_norandom_rte.npy")
    p22 = numpy.load("para_diff_random_rte.npy")
    cos_sim1 = p11.dot(p12) / (np.linalg.norm(p11) * np.linalg.norm(p12))
    cos_sim2 = p21.dot(p22) / (np.linalg.norm(p21) * np.linalg.norm(p22))
    print(cos_sim1, cos_sim2)
    #0.00030579508 -0.008579531
    #3e-4 -8.5e-3
    breakpoint()



def solve_embedding(args):
    args.load = "/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-pt[0-0]-lr0.007-seed784846687-05-16-18-05"
    model_old = ClassificationModel(args)
    _ = load_checkpoint(model_old, args) #原参数
    model_old.requires_grad_(False)
    model_old.to('cuda:0')

    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39' #rte
    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed723826066-ab-06-01-14-22'
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed141965068-90epoch-06-03-21-46/'
    model_2stepbit_step1 = ClassificationModel(args)
    _ = load_checkpoint(model_2stepbit_step1, args) #bias对但是头不对, 但我就是要bias对的,step1的输出
    model_2stepbit_step1.requires_grad_(False)
    model_2stepbit_step1.to('cuda:0')

    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed123284720-new2-05-23-21-39pretype-2step+bitfit-05-23-21-43'
    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed723826066-ab-06-01-14-22/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed723826066-ab-06-01-14-22pretype-2step+bitfit-06-01-14-43'
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed141965068-90epoch-06-03-21-46/finetune-roberta-large-rte-2step+bitfit-lr1e-05-seed141965068-90epoch-06-03-21-46pretype-2step+bitfit-06-03-22-02/'
    model_2stepbit_step2 = ClassificationModel(args)
    _ = load_checkpoint(model_2stepbit_step2, args)
    model_2stepbit_step2.requires_grad_(False)
    model_2stepbit_step2.to('cuda:0')
    #
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-all-lr1e-05-seed495704012-savemore-05-30-21-37'
    model_all = ClassificationModel(args)
    _ = load_checkpoint(model_all, args)
    model_all.requires_grad_(False)
    model_all.to('cuda:0')

    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+head-lr0.001-seed522520237-ab-pretype-2step+head-06-03-20-02'
    # args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+head-lr1e-05-seed383856847-ab-pretype-2step+head-06-03-21-24'
    args.load = '/sharefs/cogview-new/yzy/SAT/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-2step+head-lr1e-05-seed658664719-90epoch-pretype-2step+head-06-03-22-23'
    model_head = ClassificationModel(args)
    _ = load_checkpoint(model_head, args)
    model_head.requires_grad_(False)
    model_head.to('cuda:0')

    args.train_data= None
    args.valid_data=[f"hf://super_glue/{args.dataset_name}/validation"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    timers = Timers()
    data_num = len(val_data)
    val_data = iter(val_data)

    old_list = []
    bit_step1_list = []
    bit_step2_list = []
    all_list = []
    head_list = []
    from tqdm import tqdm
    for i in tqdm(range(data_num)):
        tokens, labels, attention_mask, position_ids, loss_mask = get_batch(val_data, args, timers)
        attention_output = []

        output_old = model_old(tokens, position_ids, attention_mask, attention_output = attention_output)
        output_old = output_old[0].data.cpu()
        old_list.append(output_old)

        output_step1 = model_2stepbit_step1(tokens, position_ids, attention_mask, attention_output = attention_output)
        output_step1 = output_step1[0].data.cpu()
        bit_step1_list.append(output_step1)


        output_step2 = model_2stepbit_step2(tokens, position_ids, attention_mask, attention_output = attention_output)
        output_step2 = output_step2[0].data.cpu()
        bit_step2_list.append(output_step2)


        output_all = model_all(tokens, position_ids, attention_mask, attention_output = attention_output)
        output_all = output_all[0].data.cpu()
        all_list.append(output_all)

        output_head = model_head(tokens, position_ids, attention_mask, attention_output = attention_output)
        output_head = output_head[0].data.cpu()
        head_list.append(output_head)

    breakpoint()
    old_list = torch.cat(old_list, dim=0)
    bit_step1_list = torch.cat(bit_step1_list, dim=0)
    bit_step2_list = torch.cat(bit_step2_list, dim=0)
    all_list = torch.cat(all_list, dim=0)
    head_list = torch.cat(head_list, dim=0)
    # head_list = torch.cat(head_list, axis=0)
    diff_step1 = bit_step1_list - old_list
    diff_step2 = bit_step2_list - old_list
    diff_all = all_list - old_list
    diff_head = head_list - old_list
    diff_step1_step2 = bit_step1_list - bit_step2_list
    diff_step1_all = bit_step1_list - all_list
    diff_step2_all = bit_step2_list - all_list
    breakpoint()
    diff_step1 = torch.norm(diff_step1, p=2, dim=-1).sum(dim=0)
    diff_step2 = torch.norm(diff_step2, p=2, dim=-1).sum(dim=0)
    diff_all = torch.norm(diff_all, p=2, dim=-1).sum(dim=0)
    diff_head = torch.norm(diff_head, p=2, dim=-1).sum(dim=0)
    diff_step1_step2 = torch.norm(diff_step1_step2, p=2, dim=-1).sum(dim=0)
    diff_step1_all = torch.norm(diff_step1_all, p=2, dim=-1).sum(dim=0)
    diff_step2_all = torch.norm(diff_step2_all, p=2, dim=-1).sum(dim=0)
    print(diff_step1, diff_step2, diff_all, diff_head, diff_step1_step2, diff_step1_all, diff_step2_all)


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16-1)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--ssl_load2', type=str, default=None)
    py_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    py_parser.add_argument('--dataset-name', type=str, default=None, required=True)
    #ffadd
    py_parser.add_argument('--ffadd-r', type=int, default=32)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    args.do_train = False

    # solve_ffadd(args)
    # solve_draw_diff(args)
    # calc_gradient(args)
    solve_embedding(args)
    # draw_hist()