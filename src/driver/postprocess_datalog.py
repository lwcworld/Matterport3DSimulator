import networkx as nx
import json

dir_datalog = '../../datalog/'
i_epi = 0
i_step = 20
G = nx.read_gpickle(dir_datalog+'graph_' + str(i_epi) + '_' + str(i_step) + '.gpickle')

nodes = list(G.nodes(data=True))
attr_isrobot = nx.get_node_attributes(G, 'isrobot')

# print(attr_isrobot)
print(nx.cycle_basis(G))