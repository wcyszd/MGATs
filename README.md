

### Installation

Currently, we support installation from the source.

```bash
git clone https://github.com/Graph-Learning-Benchmarks/gli.git
cd gli
pip install -e .
```
> *Note: [wget](https://www.gnu.org/software/wget/) is required to download datasets.*

To test the installation, run the following command:

```bash
python example.py --graph cora --task NodeClassification
```

The output should be something like the following:

```
> Graph(s) loading takes 0.0196 seconds and uses 0.9788 MB.
> Task loading takes 0.0016 seconds and uses 0.1218 MB.
> Combining(s) graph and task takes 0.0037 seconds and uses 0.0116 MB.
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=~/.dgl/CORA dataset. NodeClassification)**
```

### Data Loading API

To load a dataset from the remote data repository, simply use the `get_gli_dataset()` function:

```python
>>> import gli
>>> dataset = gli.get_gli_dataset(dataset="cora", task="NodeClassification", device="cpu")
>>> dataset
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)
```

Alternatively, one can also get a single graph or a list of graphs rather than a wrapped dataset by `get_gli_graph()`. Furthermore, GLI provides abstractions for various tasks (`GLITask`) and provides a function `get_gli_task()` to return a task instance. Combine these two instances to get a wrapped dataset that is identical to the previous case.

```python
>>> import gli
>>> g = gli.get_gli_graph(dataset="cora", device="cpu", verbose=False)
>>> g
Graph(num_nodes=2708, num_edges=10556,
      ndata_schemes={'NodeFeature': Scheme(shape=(1433,), dtype=torch.float32), 'NodeLabel': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={})
>>> task = gli.get_gli_task(dataset="cora", task="NodeClassification", verbose=False)
>>> task
<gli.task.NodeClassificationTask object at 0x100eff640>
>>> dataset = gli.combine_graph_and_task(g, task)
>>> dataset
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)
```

The returned dataset is inherited from `DGLDataset`. Therefore, it can be incorporated into DGL's infrastructure seamlessly:

```python
>>> type(dataset)
<class 'gli.dataset.NodeClassificationDataset'>
>>> isinstance(dataset, dgl.data.DGLDataset)
True
```
