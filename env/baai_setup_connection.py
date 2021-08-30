import json
import os
import sys

with open('/home/hostfile.json', 'r') as fin:
    t = json.load(fin)
input_txt_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_txt_path, 'w') as fout:
    ip_list = []
    for x in t:
        fout.write(x['ip'])
        fout.write(' ')
        ip_list.append(x['ip'])
sys.path.append(os.path.dirname(__file__))
from setup_connection import main
main(ip_list, 22)

