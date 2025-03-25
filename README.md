# GECC: Scalable Graph Condensation with Evolving Capabilities

# Abstract
Graph data has become a pivotal modality due to its unique ability to model relational datasets. However, real-world graph data continues to grow exponentially, resulting in a quadratic increase in the complexity of graph algorithms as graph sizes expand. Besides, although graph condensation (GC) methods have been proposed to address these scalability issues, existing approaches often treat the training set as static, overlooking the evolving nature of real-world graph data. This limitation leads to inefficiencies when condensing growing training sets.
In this paper, we introduce GECC (Graph Evolving Clustering Condensation), a scalable graph condensation method designed to handle large-scale and evolving graph data. GECC employs a clear and efficient approach by performing class-wise clustering on aggregated features. It inherits clustering centroids as the graph expands, thereby attaining an evolving capability. This methodology is supported by robust theoretical foundations and demonstrates superior empirical performance. Comprehensive experiments show that GECC achieves over 99\% of the performance of state-of-the-art graph condensation methods while delivering an impressive 1000× speedup on large datasets.

# Prepare Environments

## CUDA and PyTorch
Check [torch previous versions](https://pytorch.org/get-started/previous-versions/).
We test this repo  in  `torch 1.12.1`  with `CUDA 11.6` and `torch 2.4.0` with `CUDA 12.1`.

## Install from requirements
`pip install -r requirements.txt`
# Download Datasets

For cora, citeseer, flickr and reddit (reddit2 in pyg), the pyg code will directly download them.
For arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 
Our code will automatically download all datasets.

The default path of datasets is `../../data`.

# Reproduce
`cd graphlim`  
> Add any auguments to `bash run_<dataset>_evolve.sh` to reproduce non-evole setting results.


For small datasets

`bash run_cora_evolve.sh && bash run_cora_evolve.sh 1`

`bash run_citeseer_evolve.sh && bash run_citeseer_evolve.sh 1`

`bash run_pubmed_evolve.sh && bash run_pubmed_evolve.sh 1`

For large datasets

`bash run_flickr_evolve.sh && bash run_flickr_evolve.sh 1`

`bash run_ogbn-arxiv_evolve.sh && bash run_ogbn-arxiv_evolve.sh 1`

`bash run_ogbn-products_evolve.sh && bash run_ogbn-products_evolve.sh 1`

`bash run_reddit_evolve.sh && bash run_reddit_evolve.sh 1`

To run the whole GNN training

`python run_eval.py -D <dataset> -W -G <gpu_id>`

# Acknowledgement

Some of the algorithms are referred to paper authors' implementations and other packages.

[SimGC](https://github.com/BangHonor/SimGC)

[GCOND](https://github.com/ChandlerBang/GCond)

[GEOM](https://github.com/NUS-HPC-AI-Lab/GEOM/tree/main)
