# -*- encoding: utf-8 -*-
'''
@File    :   helpers.py
@Time    :   2023/04/10 16:54:36
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import argparse
import textwrap

def print_parser(parser, help_width=32):
    argument_list = []

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if '--help' in action.option_strings:
            continue

        arg_name = ', '.join([opt.lstrip('-') for opt in action.option_strings])
        arg_help = action.help or ''
        arg_type = action.type.__name__ if action.type else 'str'
        arg_default = str(action.default) if action.default is not None else 'None'

        argument_list.append((arg_name, arg_help, arg_type, arg_default))

    max_name_len = max([len(arg[0]) for arg in argument_list])

    print("-" * (max_name_len + 56))
    print(f"{'Argument'.ljust(max_name_len)}  Help" + " "*(help_width-4) + f"{'Type'.ljust(8)}    Default")
    print("-" * (max_name_len + 56))

    wrapper = textwrap.TextWrapper(width=help_width)

    for arg_name, arg_help, arg_type, arg_default in argument_list:
        name_str = arg_name.ljust(max_name_len)
        type_str = arg_type.ljust(8)

        wrapped_help = wrapper.wrap(arg_help)
        if not wrapped_help:
            wrapped_help = ['']

        for i, line in enumerate(wrapped_help):
            if i == 0:
                print(f"{name_str}  {line.ljust(help_width)}  {type_str}  {arg_default}")
            else:
                print(f"{''.ljust(max_name_len)}  {line.ljust(help_width)}")
        print()

def print_aligned_string_list(str_list, column_spacing=2):
    # 获取字符串列表中的最长字符串长度
    max_length = max(len(s) for s in str_list)

    # 计算终端宽度以便我们知道多少列可以显示
    try:
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        terminal_width = 80
    print("-" * (terminal_width-5))

    # 计算每行可容纳的列数
    columns_per_row = (terminal_width + column_spacing) // (max_length + column_spacing)

    # 计算需要多少行来显示所有字符串
    rows_required = (len(str_list) + columns_per_row - 1) // columns_per_row

    # 按行打印字符串，确保对齐
    for row in range(rows_required):
        line = ""
        for col in range(columns_per_row):
            index = row + col * rows_required
            if index < len(str_list):
                line += str_list[index].ljust(max_length + column_spacing)
        print(line.strip())
    print("-" * (terminal_width-5))

def list_avail_models():
    from .model import official
    model_list = []
    for name in dir(official):
        # if is a class
        if isinstance(getattr(official, name), type):
            model_list.append(name)
    print('Available model definitions (example: "from sat import GLMModel"):')
    print_aligned_string_list(model_list)
    return model_list

def list_avail_pretrained():
    from sat.resources.urls import MODEL_URLS
    # iterate over all pretrained models into a list
    model_list = []
    for model_name, model_url in MODEL_URLS.items():
        model_list.append(model_name)
    print('Available pretrained models (example: sat.AutoModel.from_pretrained("roberta-base")):')
    print_aligned_string_list(model_list)
    return model_list