import os
import torch
import argparse
from SwissArmyTransformer import get_args, AutoModel
# from SwissArmyTransformer.model.official.bert_model import BertModel

args = get_args()

model_type = 'cogview-base'
model, args = AutoModel.from_pretrained(args, model_type)