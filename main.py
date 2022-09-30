
import pandas_datareader as web
import pandas as pd
import numpy as  np
import networkx as nx



df  =  web.DataReader('ETH-USD', 'yahoo')

n = np.arange(1, 15)
print(n)
bic = np.zeros(n.shape)
print(n.shape)
df.reset_index(inplace=True)


print(df)




data = df[['Date','High', 'Low', 'Open', 'Adj Close', 'Volume']]


data2  = pd.DataFrame(data)



data2['pct_chg'] = data2['Adj Close'].pct_change()

print(data2.head())

data2['state'] = data['pct_chg'].apply(lambda x: 'up' if (x > 0.001)\
    else ('down' if (x < -0.001)\
    else 'no_change'))
print(data2.tail())

data2['prev_state'] = data2['state'].shift(1)
print(data2.tail())

state_space = data2[['prev_state', 'state']]
state_space_matrix = data2.groupby(['prev_state',  'state']).size().unstack()

print(state_space_matrix.sum())

transition_matrix = state_space_matrix.apply(lambda x: x/float(x.sum()), axis=1)
print(transition_matrix)
#VERIFY MATRIX IS  LEGIT

print(transition_matrix.sum(axis=1))


def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
            return edges
edges_wts = _get_markov_edges(transition_matrix)
print(edges_wts)



states = ['up', 'down', 'no_change']
G = nx.MultiDiGraph()
# nodes correspond to states
G.add_nodes_from(state_space_matrix)
print(f'Nodes:\n{G.nodes()}\n')
# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    print(f'Edges:')
    print(G.edges(data=True))
    pos = nx.drawing.nx_pydot.graphviz_layout(G,)
    nx.draw_networkx(G, pos)
    edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)  
    nx.drawing.nx_pydot.write_dot(G, 'sup.dot')

#create  markov  edges
