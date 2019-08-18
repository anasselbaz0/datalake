import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# Declare abstract edges
N = 8
node_indices = np.arange(N, dtype=np.int32)
source = np.zeros(N, dtype=np.int32)
target = node_indices
print(source, target)
simple_graph = hv.Graph(((source, target),))
simple_graph
