import sys
import random as rand

if len(sys.argv) != 5:
    exit(1)


filename = sys.argv[1]
n_nodes = int(sys.argv[2])
n_nets = int(sys.argv[3])
ratio = float(sys.argv[4])
print(f'[Info] Filename: {filename}')
print(f'[Info] #Nodes: {n_nodes}')
print(f'[Info] #Nets: {n_nets}')
print(f'[Info] Ratio: {ratio}')

max_net_deg = 10
min_net_deg = 2

node_names = [ f'a{i}' for i in range(n_nodes) ]
net_names = [ f'n{i}' for i in range(n_nets) ]

def gen_one_net(deg):
    if deg > n_nodes:
        deg = n_nodes
    return rand.sample(range(0, n_nodes), deg)

nets = [
    gen_one_net(rand.randint(min_net_deg, max_net_deg)) for _ in range(n_nets)
]

with open(filename, 'w') as f:
    f.write(f'{n_nodes}\n')
    f.write(f'{n_nets}\n')
    for i, net in enumerate(nets):
        f.write(f'{net_names[i]}')
        for j in net:
            f.write(f' {node_names[j]}')
        f.write('\n')
    f.write(f'{ratio}\n')

