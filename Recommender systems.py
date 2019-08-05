import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import defaultdict
from surprise import SVD
from surprise.model_selection import cross_validate
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import random
import time
import warnings
warnings.filterwarnings('ignore')
start_time = time.time()
tran = pd.read_csv("clickStream.csv")
click =tran[['category', 'uuid','price','date']]
u_cat = click['category'].unique()
map_cat = []
for i in range(0,302):
    map_cat.append(i)
map_cat_dict = dict(zip(u_cat, map_cat))    
mapping=click['category'].map(map_cat_dict)
click['category'] = mapping
click['date']=pd.to_datetime(click.date)
no_of_Purchase_per_user = click.groupby(by='uuid')['category'].count().sort_values(ascending=False)
train_sparse_matrix = sparse.csr_matrix((click.price.values,(click.uuid.values,click.category.values)))

def get_average_purchase(sparse_matrix, of_users):
    ax = 1 if of_users else 0 
    sum_of_price = sparse_matrix.sum(axis=ax).A1
    is_rated = sparse_matrix!=0
    no_of_price_counts = is_rated.sum(axis=ax).A1
    u,m = sparse_matrix.shape
    average_price = { i : sum_of_price[i]/no_of_price_counts[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_price_counts[i] !=0}
    return average_price

train_averages = dict()
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages['user'] = get_average_purchase(train_sparse_matrix, of_users=True)
train_averages['category'] =  get_average_purchase(train_sparse_matrix, of_users=False)

def compute_user_similarity(sparse_matrix, compute_for_few=False, top = 100, verbose=False, verb_for_n_rows = 20,
                            draw_time_taken=True):
    no_of_users, _ = sparse_matrix.shape
    row_ind, col_ind = sparse_matrix.nonzero()
    row_ind = sorted(set(row_ind))
    time_taken = list()
    rows, cols, data = list(), list(), list()
    temp = 0
    for row in row_ind[:top] if compute_for_few else row_ind:
        temp = temp+1
        sim = cosine_similarity(sparse_matrix.getrow(row), sparse_matrix).ravel()
        top_sim_ind = sim.argsort()[-top:]
        top_sim_val = sim[top_sim_ind]
        rows.extend([row]*top)
        cols.extend(top_sim_ind)
        data.extend(top_sim_val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(no_of_users, no_of_users)), time_taken      

u_u_sim_sparse, _ = compute_user_similarity(train_sparse_matrix, compute_for_few=True, top = 100,verbose=True)
clickstream_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=15)
trunc_svd = clickstream_svd.fit_transform(train_sparse_matrix)
trunc_matrix = train_sparse_matrix.dot(clickstream_svd.components_.T)
trunc_sparse_matrix = sparse.csr_matrix(trunc_matrix)
trunc_u_u_sim_matrix, _ = compute_user_similarity(trunc_sparse_matrix, compute_for_few=True, top=50, verbose=True, 
                                                 verb_for_n_rows=10)
c_c_sim_sparse = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)
cat_ids = np.unique(c_c_sim_sparse.nonzero()[1])
similar_categories = dict()
for movie in cat_ids:
    sim_categories = c_c_sim_sparse[movie].toarray().ravel().argsort()[::-1][1:]
    similar_categories[movie] = sim_categories[:100]
all_cat = pd.DataFrame(u_cat)

def similar_cat(ip):
        mv_id = ip
        idx_list = []
        print("\nCATEGORY ----->",all_cat.loc[mv_id].values[0])

        print("\nIt has {} Purchase from users.".format(train_sparse_matrix[:,mv_id].getnnz()))

        print("\nIt has {} Possible Categories user can purchase, the Top 10 Categories are..".format(c_c_sim_sparse[:,mv_id].getnnz()))
        print("\n")
        similarities = c_c_sim_sparse[mv_id].toarray().ravel()

        similar_indices = similarities.argsort()[::-1][1:]

        similarities[similar_indices]

        sim_indices = similarities.argsort()[::-1][1:]
        result_df = all_cat.loc[sim_indices[:10]][0]
        result_df.columns = ['SIMILAR CATEGORIES']
        idx_list = list(result_df.index.values)
        se = pd.Series([1,2,3,4,5,6,7,8,9,10],name="rank",index = idx_list)
        rank_results = pd.concat([result_df, se], axis=1)
        rank_results.columns = ['SIMILAR CATEGORIES','SIMILARITY RANK']
        #result_df['SIMILARITY RANK'] = se.values
        #result_df['SIMILARITY RANK'] = rank
        print(rank_results)

def predict_chk(li):
    recom = []
    for i in li:
        temp = []
        mv_id = i
        print("-"*50)
        print("\nCATEGORY ----->",all_cat.loc[mv_id].values[0])

        print("\nIt has {} Purchase from users.".format(train_sparse_matrix[:,mv_id].getnnz()))

        print("\nIt has {} Possible Categories user can purchase..".format(c_c_sim_sparse[:,mv_id].getnnz()))
        print("-"*50)
        similarities = c_c_sim_sparse[mv_id].toarray().ravel()

        similar_indices = similarities.argsort()[::-1][1:]

        similarities[similar_indices]

        sim_indices = similarities.argsort()[::-1][1:]
        temp=list(all_cat.loc[sim_indices[:3]][0])
        recom.extend(temp)
    return recom

def user_past_purchase(u_id):
    try:
        user_info = click[(click['uuid']==u_id)]
        u_list = list(user_info['category'])
        print("PAST PURCHASE HISTORY OF THE USER ",u_id," ARE:\n")
        for i in u_list:
            print(all_cat.loc[i].values[0])
        print("\nPurchase Details : ")
        result = predict_chk(u_list)
        result_df = pd.DataFrame(result)
        result_df.columns=['categoreis']
        print("\nFuture PURCHASE PREDICTION OF THE USER ",u_id," ARE:")
        print("\n")
        print("Top Predicted Categories for future purchase along with Normalized Probablity Scores are : \n")
        print(result_df['categoreis'].value_counts(normalize=True,sort=False))
    except:
        print("Please enter correct User Id (uuid)\n")

choice=1
while(choice!=3):
    print("       1)USER--->CATEGORY PURCHASE PREDICTION         ")
    print("       2)CATEGORY--->CATEGORY PURCHASE PREDICTION     ")
    print("       3)EXIT                                         ")
    print("   Enter your choice (1 or 2 or 3)  ")
    try:
        choice = int(input())
        if(choice==1):
            print("NEXT CATEGORY PURCHASE PREDICTION OF THE USER\n")
            print("Enter User_Id of the user (uuid) \n")
            uu_id = int(input())
            user_past_purchase(uu_id)
            print("--- %s seconds ---" % (time.time() - start_time))
        elif(choice==2):
            print("SEARCH SIMILAR CATEGORIES THAT ANY USERS CAN PURCHASE FROM : \n")
            print("ENTER THE CATEGORY : \n")
            search_cat = input()
            try:
                search_key = map_cat_dict[search_cat]
                similar_cat(search_key)
            except:
                print("Please Enter correct category\n")
            print("--- %s seconds ---" % (time.time() - start_time))
        elif(choice==3):
            break
        else:
            print("PLEASE ENTER OPTION 1 OR 2")
    except:
        print("PLEASE CORRECT ENTER OPTION 1 OR 2")
        

