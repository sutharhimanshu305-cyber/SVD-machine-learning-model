# SVD-machine-learning-model
SVD (Singular Value Decomposition) is a mathematical technique used to decompose a matrix into three matrices — U, Σ, and Vᵀ — revealing important patterns in data. In Machine Learning, SVD is widely used for dimensionality reduction, noise reduction, and recommendation systems (like in Netflix or Amazon).
SVD (Singular Value Decomposition)
It is a way to break a big matrix into 3 smaller parts (matrix) that are easier to understand and work with.

A = U x E x V^T

A - complete matrix (original matrix)
U - Left Singular vector (People's Preferenece --> Who like what)
E - sigma - Singular value (Strenght of the pattern) --> how strong each pattern is
V^T - (v transpose) - Right singualar vector (Movie features) -> What kind of movie it is
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# Rows - Users
# columns - Movies

rating_data = {
    'movie1': [5,4,np.nan,2,1],
    'movie2': [3,np.nan,4,3,1],
    'movie3': [4,np.nan,5,np.nan,1],
    'movie4': [np.nan,2,3,4,2],
    'movie5': [1,2,np.nan,5,np.nan]
}

rating_data
output: {'movie1': [5, 4, nan, 2, 1],
 'movie2': [3, nan, 4, 3, 1],
 'movie3': [4, nan, 5, nan, 1],
 'movie4': [nan, 2, 3, 4, 2],
 'movie5': [1, 2, nan, 5, nan]}

 
rating_df = pd.DataFrame(rating_data,index = ['User1','User2','User3','User4','User5'])
rating_df
output:movie1	movie2	movie3	movie4	movie5
User1	5.0	3.0	4.0	NaN	1.0
User2	4.0	NaN	NaN	2.0	2.0
User3	NaN	4.0	5.0	3.0	NaN
User4	2.0	3.0	NaN	4.0	5.0
User5	1.0	1.0	1.0	2.0	NaN

# 1. We will find out each user's mean rating
user_mean = rating_df.mean(axis = 1)
user_mean
output: 
0
User1	3.250000
User2	2.666667
User3	4.000000
User4	3.500000
User5	1.250000

dtype: float

# 2. Demean the rating (we will substract user mean)---
# Null values (means user has not rated that movie yet)
demeaned_ratings = rating_df.sub(user_mean, axis = 0).fillna(0)
demeaned_ratings # This is my matrix A
output: 	movie1	movie2	movie3	movie4	movie5
User1	1.750000	-0.25	0.75	0.000000	-2.250000
User2	1.333333	0.00	0.00	-0.666667	-0.666667
User3	0.000000	0.00	1.00	-1.000000	0.000000
User4	-1.500000	-0.50	0.00	0.500000	1.500000
User5	-0.250000	-0.25	-0.25	0.750000	0.000000

demeaned_ratings.values
output: array([[ 1.75      , -0.25      ,  0.75      ,  0.        , -2.25      ],
       [ 1.33333333,  0.        ,  0.        , -0.66666667, -0.66666667],
       [ 0.        ,  0.        ,  1.        , -1.        ,  0.        ],
       [-1.5       , -0.5       ,  0.        ,  0.5       ,  1.5       ],
       [-0.25      , -0.25      , -0.25      ,  0.75      ,  0.        ]])

       

 
