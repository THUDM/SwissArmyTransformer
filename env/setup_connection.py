# -*- encoding: utf-8 -*-
'''
@File    :   setup_connection.py
@Time    :   2021/01/16 16:50:36
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

def main(ip_list, port=2222):
    ssh_config = ''
    line_format = 'Host node{}\n\tUser root\n\tPort {}\n\tHostname {}\n'
    for i, ip in enumerate(ip_list):
        ssh_config += line_format.format(i, port, ip)
    
    ret = os.system(f'echo \"{ssh_config}\" > ~/.ssh/config && chmod 600 ~/.ssh/config')
    assert ret == 0

    hostfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hostfile')
    with open(hostfile_path, 'w') as fout:
        for i, ip in enumerate(ip_list):
            fout.write(f'node{i} slots=8\n')
    print(f'Successfully generating hostfile \'{hostfile_path}\'!')

if __name__ == "__main__":
    main(sys.argv[1:])