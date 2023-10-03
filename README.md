# FS_MGNNSHC
Code of paper Feature Selection of Microarray Data Using Multidimensional Graph Neural Network and Supernode Hierarchical Clustering 

For reviewers to review

The current version of the code is not fully curated. We present the core code and examples for reviewers to review. The full code will be published with the final version of the paper.

You can achieve this by running the main.py script, which creates a graph-structured dataset with two types of relationships to perform feature selection and classification tasks.

![fig1](https://github.com/xwdshiwo/FS_MGNNSHC/assets/35399345/960bc131-84de-4d03-b437-3265598c3a11)


Files：

**GraphPurification.py**: This module purifies redundant relationships within the graph.

**MultiGraphFilter.py**: Uses multidimensional graph structure relationships for node information propagation and aggregation.

**LinkPrediction.py**: This script supplements potential feature relationships not covered or predicted in GeneMANIA.

**MultidimensionalNodeEvaluator.py**: Uses multidimensional feature evaluation methods to produce node rankings.

**SupernodeDiscovery.py**: Utilizes spectral clustering techniques to cluster the graph, aggregate redundant features, and combine with node rankings for preliminary feature filtering.

**GraphPooling.py**: Employs graph pooling models for classification tasks. By examining pooling parameters, it further assesses node importance, providing deeper insights into feature evaluation.

**baselineMethods**: Contains feature selection methods for comparison.

**Example:**

Upon execution, the program processes the graph and outputs rankings from the multidimensional feature evaluation and clustering information:

**feature evaluator result: [24 21 0 27 26 16 22 23 5 28 18 11 17 8 3 20 19 1 12 14 6 9 10 2 29 4 7 25 13 15]

cluster result: [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]**

Based on this information, we can filter the optimal nodes from each cluster for further graph pooling tasks. The graph pooling will establish a further classification model and output additional node rankings:

**Node ranks after second pooling: tensor([ 0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8])**




