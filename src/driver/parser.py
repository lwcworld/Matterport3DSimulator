import networkx as nx







G = nx.read_gpickle('datalog_roomnav/test1/graph_0_43.gpickle')

# G = nx.read_gpickle('/home/lwcubuntu/Matterport3DSim/Matterport3DSimulator/datalog_roomnav/test1/graph_0_0.gpickle')
# att_isrobot = nx.get_node_attributes(G, 'isrobot')
# att_isopen = nx.get_node_attributes(G, 'isopen')
# att_ix = nx.get_node_attributes(G, 'ix')
#
# print(att_isrobot)