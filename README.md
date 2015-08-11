# Topic Modeling
Topic models are a suite of algorithms that uncover the hidden thematic structure in document collection. These algorithms help us develop new ways to search,browse and summarize large archivers of texts.

We include below algorithms of topic model in this package:
1) Online LDA
   Adapted from MLlib Online LDA implementation(PR#4419)

2) Gibbs sampling LDA
   Adapted from Spark PRs(#1405, #4807) and JIRA SPARK-5556 discussions with extension predict
   https://github.com/witgo/spark/tree/lda_Gibbs
   https://github.com/EntilZha/spark/tree/LDA-Refactor
   https://github.com/witgo/zen/tree/lda_opt/ml

3) Online hierarchical Dirichlet process
   Reference Online HDP paper: 
    Chong Wang John Paisley David M. Blei, "Online Variational Inference for the Hierarchical Dirichlet Process." 
      
