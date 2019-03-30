"""
Created on 2/28/19

@author: ivanchen

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightfm.evaluation import precision_at_k,auc_score
import scipy
import time
from IPython.display import Markdown, display


def train_test_split(ratings, split_count, fraction=None):
    """
    Split recommendation data into train and test sets

    Params
    ------
    ratings : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = ratings.copy().tocoo()
    test = scipy.sparse.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int32(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                   'interactions for fraction of {}') \
                  .format(2 * split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_ratings] = 0.
        # These are just 1.0 right now
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

def printmd(string):
    display(Markdown(string))
    
def sample_train_recommendation(model, train, data_meta, user_ids, k, name, mapping, tag=None, user_features=None,
                                item_features=None, num_threads=2):
    n_users, n_items = train.shape

    # =============================================================================
    #     ranks = model.predict_rank(interactions,
    #                                train_interactions=train_interactions,
    #                                user_features=user_features,
    #                                item_features=item_features,
    #                                num_threads=num_threads,
    #                                check_intersections=check_intersections,
    #                                )
    #
    #     ranks.data = np.less(ranks.data, k, ranks.data)
    #
    #     precision = np.squeeze(np.array(ranks.sum(axis=1))) / k
    #
    #     if not preserve_rows:
    #         precision = precision[test_interactions.getnnz(axis=1) > 0]
    #
    #     return precision
    # =============================================================================

    for user_id in user_ids:

        t_idx = {value: key for key, value in mapping.items()}
        u_idx = [x for x in train.tocsr()[user_id].indices]
        known_positives = data_meta.loc[u_idx, name]  # may need change
        if tag is not None:
            known_tags = data_meta.loc[u_idx, tag]  # get item tags.

        if (len(known_positives) < k):
            print('not enough known positives, return max number')

        scores = model.predict(user_id, np.arange(n_items), user_features=user_features, item_features=item_features,
                               num_threads=num_threads)
        i_idx = [x for x in np.argsort(-scores)]
        top_items = data_meta.loc[i_idx, name]
        if tag is not None:
            top_tags = data_meta.loc[i_idx, tag]  # get item tags.

        printmd("**User %s**" % user_id)
        printmd("**Known positives:**")

        if tag is not None:
            for x in range(len(known_positives)):
                print(" %s | %s" % (known_positives.values[x], known_tags.values[x]))
        else:
            for x in known_positives[:len(known_positives)]:
                print("        %s" % x)

        printmd("**Recommended:**")
        cnt = 0
        if tag is not None:
            for x in range(k):
                print(" %s | %s" % (top_items.values[x], top_tags.values[x]))
                if (top_items.values[x] in known_positives.values):
                    cnt += 1
                    print('This one clicked')
        else:
            for x in top_items[:k]:
                print("        %s" % x)
                if (x in known_positives.values):
                    cnt += 1
                    print('This one clicked')
        #printmd('*cnt: *' + str(cnt))
        printmd('*k_p: %s*'%str(len(known_positives)))
        p_k = cnt / k
        printmd('*precicion at k : %s*'%str(p_k))
        print('----------------------------------------------------------------------')


def sample_test_recommendation(model, train, test, data_meta, user_ids, k, name, mapping, tag=None,
                               train_interactions=None, user_features=None,
                               item_features=None, num_threads=2):
    n_users, n_items = test.shape

    for user_id in user_ids:
        
        printmd("**User %s**" % user_id)
        
        t_idx = {value: key for key, value in mapping.items()}
        u_idx = [x for x in test.tocsr()[user_id].indices]

        known_positives = data_meta.loc[u_idx, name]  # may need change

        print('length of known_positives: ' + str(len(known_positives)))
        if (len(known_positives) == 0):
            sample_train_recommendation(model, train, data_meta, [user_id], k, name, mapping, tag, user_features,
                                        item_features)
            continue

        elif (len(known_positives) < k):
            print('not enough known positives, return max number')

        if tag is not None:
            known_tags = data_meta.loc[u_idx, tag]  # get item tags.

        if (train_interactions is None):
            scores = model.predict(user_id, np.arange(n_items), user_features=user_features,
                                   item_features=item_features,
                                   num_threads=num_threads)
            i_idx = [x for x in np.argsort(-scores)]
            top_items = data_meta.loc[i_idx, name]
            if tag is not None:
                top_tags = data_meta.loc[i_idx, tag]  # get item tags.

        else:
            item_ids = np.delete(np.arange(n_items), train.tocsr()[user_id].indices)
            scores = model.predict(user_id, item_ids, user_features=user_features, item_features=item_features,
                                   num_threads=num_threads)
            i_idx = [x for x in np.argsort(-scores)]
            top_items = data_meta.loc[i_idx, name]
            if tag is not None:
                top_tags = data_meta.loc[i_idx, tag]  # get item tags.

        
        printmd("**Known positives:**")

        if tag is not None:
            for x in range(len(known_positives)):
                print(" %s | %s" % (known_positives.values[x], known_tags.values[x]))
        else:
            for x in known_positives[:len(known_positives)]:
                print("        %s" % x)

        printmd("**Recommended:**")
        cnt = 0
        if tag is not None:
            for x in range(k):
                print(" %s | %s" % (top_items.values[x], top_tags.values[x]))
                if (top_items.values[x] in known_positives.values):
                    cnt += 1
                    print('This one clicked')
        else:
            for x in top_items[:k]:
                print("        %s" % x)
                if (x in known_positives.values):
                    cnt += 1
                    print('This one clicked')
        #printmd('*cnt: *' + str(cnt))
        printmd('*k_p: %s*'%str(len(known_positives)))
        p_k = cnt / k
        printmd('*precicion at k : %s*'%str(p_k))
        print('----------------------------------------------------------------------')
        



def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-' * spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing - 2)
        bottom += '+{}'.format('=' * spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


def patk_learning_curve(model, train, test,
                        iterarray, user_features=None,
                        item_features=None, k=5,
                        **fit_params):
    old_epoch = 0
    train_patk = []
    test_patk = []
    warp_duration = []
    #    bpr_duration = []
    train_warp_auc = []
    test_warp_auc = []
    #   bpr_auc = []
    headers = ['Epoch', 'train p@5', 'train_auc', 'test p@5', 'test_auc']
    print_log(headers, header=True)
    for epoch in iterarray:
        more = epoch - old_epoch
        start = time.time()
        model.fit_partial(train, user_features=user_features,
                          epochs=more, item_features=item_features, **fit_params)
        warp_duration.append(time.time() - start)
        train_warp_auc.append(auc_score(model, train, item_features=item_features).mean())
        test_warp_auc.append(auc_score(model, test, item_features=item_features, train_interactions=train).mean())
        this_test = precision_at_k(model, test, train_interactions=train, item_features=item_features, k=k)
        this_train = precision_at_k(model, train, train_interactions=None, item_features=item_features, k=k)

        train_patk.append(np.mean(this_train))
        test_patk.append(np.mean(this_test))
        row = [epoch, train_patk[-1], train_warp_auc[-1], test_patk[-1], test_warp_auc[-1]]
        print_log(row)
    return model, train_patk, test_patk, warp_duration, train_warp_auc, test_warp_auc


def get_user_index(test):
    return scipy.sparse.find(test)[0]


def extract_topic_name(topic):
    if (isinstance(topic, list)):
        topics = []
        for label in topic:
            topics.append(label['name'])
        return topics

    else:
        raise TypeError("zhihu topics should be a list")


def plot_patk(iterarray, patk,
              title, k=5):
    plt.plot(iterarray, patk);
    plt.title(title, fontsize=20);
    plt.xlabel('Epochs', fontsize=24);
    plt.ylabel('p@{}'.format(k), fontsize=24);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);


def plot_vec(epoch, vec, ylabel):
    plt.plot(epoch, np.array(vec))
    plt.xlabel('Epochs', fontsize=14);
    plt.ylabel(ylabel, fontsize=14);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);


def get_similar_tags(model, tag_id):
    # Define similarity as the cosine of the angle
    # between the tag latent vectors

    # Normalize the vectors to unit length
    tag_embeddings = (model.item_embeddings.T
                      / np.linalg.norm(model.item_embeddings, axis=1)).T

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar