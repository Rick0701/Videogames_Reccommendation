# Videogames recommendations and gamers segmentation

**Contributors**

*Riccardo Ricci* ([https://github.com/Rick0701](https://github.com/Rick0701))

*Royce Lam* ([https://github.com/rollroyces](https://github.com/rollroyces))

*Candace Lei* ([https://github.com/picandace](https://github.com/picandace))

**Short Intro**
Developed recommendation models and segmentation techniques using a large-scale Steam dataset (41M+ records). Implemented supervised learning approaches (popularity-based, collaborative and content filtering, kNN, matrix factorization, neural collaborative filtering with Keras) to predict game preferences, with deep learning delivering the best performance. Applied unsupervised methods (PCA, clustering) to group gamers by behavior and preferences, supporting targeted recommendations and marketing insights. Tools included Python (Pandas, Keras), PCA/cluster analysis, and visualization with Matplotlib.

**Main Tools**

<div align="center">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python" />
<img src='https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white' alt='Pandas' />
<img src='https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white' alt='Keras' />
</div>

Other Tools: matplotlib, Principal Component Analysis, Cluster Analysis, 

### Motivation

There are many video games to play but how to choose which to play. Recommendation systems come to the rescue helping gamers to know which games to play next and maximizing their satisfaction. Another interesting and related topic is allowing similar gamers to be connected based on similar interests as well as helping video gaming companies to provide better customer targeting
Thus, the objective of this project is **recommending videogames based on games and gamers similarity (supervised learning) and segmenting gamers into groups of similar preferences (unsupervised learning)**

### Data Sources
The datasets are retrieved from Kaggle, Kozyriev (2023). It consists of over 41 million cleaned and preprocessed data about users, games and recommendations scraped from the popular game store Steam ([https://store.steampowered.com/](https://store.steampowered.com/)). The datasets, available as CSV files,  are the following (underlined are joinable fields):
- The *user dataset* contains anonymised *user_id*, *number of purchased games* and *reviews by users*. 
- The *game dataset* contains *game_id, title, release date, supported platforms* (e.g. Windows, Mac), *rating* and *price*. There is an additional dataset with *games metadata* containing *description* and *tags* (e.g. Action, Adventure). 
- The *recommendation dataset* contains *game_id, user_id, review_id, recommended* (true or false), *number of hours played, review date, the number of users who found the review helpful and funny*.   

### Methodology

To reccommend videogames, we adopted different supervised learning techniques with the aim of testing these methods as well as finding the best method for games recommendation:
- Popularity-based model
- Content-based filtering
- Collaborative-based filtering
    - kNN
    - Matrix Factorization
    - Deep learning (Neural Collaborative Filtering)

For gamers segmentation, we opted for Principal Component Analysis and Cluster Analysis.

### Findings

**Supervised Learning**

Our results show that the Deep Learning model is the best model in terms of metrics evaluation. This result provides support to the current literature (He et al., 2017). We discovered that the model is sensitive to the number of neurons. Critically, we discovered that the optimal number depends on the data volume. Yet, this model can be particularly complex and heavy to train. In some situations, where easiness to explain is key especially with practitioners and low data volume, approaches like k-NN and Matrix Factorization provided valuable alternatives.

Although not shown above, we also checked that there is a trade-off between the amount of data used in training and loss convergence. Increasing the size of the training data determined an increase in the amount of time to reach convergence, and in specific cases convergence never occurs. We opted for a balance between time and data needed for convergence and amount of training data for better generalization on new unseen data.

**Unsupervised learning** 

We recognized that it is heavily dependent on humans to correctly interpret components and clusters. As a result, it can be less robust compared to supervised learning. The best approach is to integrate these models e.g. using PCA to find a set of manageable game tags and then applying a Neural Network along with. PCA helps reduce the curse of dimensionality providing a great support to supervised learning techniques. We aim to experiment with this hybrid approach in the future.

On our hierarchical clustering, surprisingly we acknowledged that it is highly dependent on the number of users. We tried with a higher number of users (greater than 10000 users) but the kernel interrupted most of the times. The clustering assignment can be further improved by applying iterative approaches like k-means clustering. Knowing the number of clusters and the clustering centers from the hierarchical clustering, we could evaluate a better k-means clustering. In this respect, we could evaluate using metrics like silhouette score or within-cluster sum of squares. 

### Challenges, Lesson Learned and Future Work

- ***Data size***. We faced several challenges related to the size of our data which slowed our analysis. We decided only at later stages to filter out data. 
Lesson learned: starting with small data and increasing later when the initial exploration is done and a model is stable. In particular, the scalability of memory-based k-NN in collaborative filtering is often limited by the size of the data. 
As the number of users and items in the dataset increases, the computational resources required to calculate and store the similarity matrix also increase significantly. Additionally, calculating the cosine similarity between user vectors has been computationally expensive. As the number of users increases, the time required to find the K nearest neighbors for each user also grows, making the algorithm less efficient. To address these limitations future work could investigate nearest neighbor search algorithms, such as locality-sensitive hashing (LSH) or tree-based methods like KD-trees or ball trees. These techniques allow for efficient retrieval of nearest neighbors without explicitly calculating the similarity between all pairs of users. Furthermore, distributed computing frameworks, such as Apache Spark or Hadoop, can be employed to parallelize the computation and handle larger datasets. By distributing the workload across multiple machines, it becomes feasible to process and analyze big data in collaborative filtering.

- ***Code Efficiency***. We spent several days building the user-game matrix while recognizing at later stages that there were some data inconsistencies. Forced to build the user-game matrix in a short time, we had to find easy and rapid solutions. We looked at similar work (He et al., 2017) and documentation on sparse matrices and we found a much easier way. 
Lesson learned: when the code becomes too complicated it is very often not the right path and very likely to lead to error. In other words, in such situations, there is often another more simple way to do it. Look at previous work and study the documentation.

- ***Data testing***. Another challenge was related to data inconsistency which wrongly led us to believe that the model was not tuned perfectly. We spent several hours tuning the hyper parameter while recognizing, only at later stages, there were some inconsistencies in the data that affected the model training. 
Lesson learned: It is really important to test the data before initiating the model training and hyperparameter tuning

- ***Limited Number of Features and Hybrid Models***. Constrained to a user-game matrix composed of 1s and 0s, it was not easy to integrate other query features such as game tags. In fact, the models proposed in this paper do not simultaneously consider user-game interaction and users or game features. Adding side features is not easy but the deep learning framework offers interesting insights in this respect (He et al., 2017). Future work could add, for instance, games tags along with the games embedding matrix (see the Appendix). Summarizing, incorporating other types of information, such as item features or contextual information, into the recommendation process can further enhance the accuracy and relevance of recommendations.  Future work could also experiment with hybrid approaches that combine memory-based methods with model-based or deep learning techniques (Moreira, 2020; He et al. 2027).



### References

Kozyriev, A. (2023). Game Recommendations on Steam [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/2871694

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182)

Malaeb, M., (2017) https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54

Moreira, G. (2020) https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101#Collaborative-Filtering-model 

Maklin, C. (2022) https://medium.com/@corymaklin/model-based-collaborative-filtering-svd-19859c764cee 

Nguyen, L. V., Vo, Q. T., & Nguyen, T. H. (2023). Adaptive KNN-Based Extended Collaborative Filtering Recommendation Services. Big Data and Cognitive Computing, 7(2), 106.
