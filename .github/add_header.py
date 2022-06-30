g = open("README_tmp.md", "w", encoding='utf-8')

g.write("# SwissArmyTransformer Documentation\n")

with open("README.md", "r", encoding='utf-8') as f:
    for line in f:
        g.write(line)

g.close()