import numpy as np
from tqdm import tqdm


def get_train_test(data, have_test=False):
    """
    Split the data into a training and a test set.
    Args:
        data (pandas DataFrame) : the data to split
        have_test (bool) : whether to split the data into a training and a test set or not
    Returns:
        R_train (numpy array) : the training set
        R_test (numpy array) : the test set (None if have_test is False)
    """
    user_id = data['user_id'].unique()
    item_id = data['book_id'].unique()

    user_map = {u: i for i, u in enumerate(user_id)}
    item_map = {b: j for j, b in enumerate(item_id)}

    if have_test: # Split the data into a training and a test set
        nb_eval_user = data['user_id'].value_counts()
        nb_eval_item = data['book_id'].value_counts()

        #Get the list of pairs (user, item) that have been evaluated at least 5 times to avoid cold start
        list_pairs = [(u, b) for u, b in zip(data['user_id'], data['book_id']) if
                      nb_eval_user[u] > 5 and nb_eval_item[b] > 5]

        books_test = []
        users_test = []
        test_data = []
        for u, b in list_pairs:
            if u not in users_test and b not in books_test: # We add the first evaluation of each user/item pair to the test set
                test_data.append(data[(data['user_id'] == u) & (data['book_id'] == b)].index[0])
                users_test.append(u)
                books_test.append(b)

        test = data.loc[test_data]
        train = data.drop(test_data)
    else:
        test = None
        train = data

    user_id_train = train['user_id'].unique()
    item_id_train = train['book_id'].unique()

    # Check that all users/items in the full set are in the training set
    if set(user_id_train) != set(user_id) or set(item_id_train) != set(item_id):
        print("Warning: not all users/items are in the training set")
        print("Users in full set but not in training set:", set(user_id) - set(user_id_train))
        print("Items in full set but not in training set:", set(item_id) - set(item_id_train))

    print("Size of full set:", data.shape)
    print("Size of training set:", train.shape)
    print("Size of test set:", test.shape if have_test else None)

    # Fill the matrix R_train with the ratings from the training set
    R_train = np.full((len(user_id), len(item_id)), np.nan)
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Filling matrix R_train"):
        R_train[user_map[row['user_id']], item_map[row['book_id']]] = row['rating']
    
    # Fill the matrix R_test with the ratings from the test set
    R_test = None
    if have_test:
        R_test = np.full((len(user_id), len(item_id)), np.nan)

        for _, row in tqdm(test.iterrows(), total=len(test), desc="Filling matrix R_test"):
            R_test[user_map[row['user_id']], item_map[row['book_id']]] = row['rating']

    return R_train, R_test