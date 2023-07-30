import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import StratifiedKFold
def split_data(y, k_shot):
    num_classes = y.max() + 1
    all_indices = np.arange(len(y))

    num_val = int(len(all_indices) * 0.2)
    num_test = int(len(all_indices) * 0.2)

    val_indices = np.random.choice(all_indices, num_val, replace=False)
    all_indices = np.setdiff1d(all_indices, val_indices)

    test_indices = np.random.choice(all_indices, num_test, replace=False)
    remaining_indices = np.setdiff1d(all_indices, test_indices)

    train_indices = []

    if k_shot >= 1:
        k_shot = int(k_shot)
        for i in range(num_classes):
            class_indices = np.isin(remaining_indices, np.where(y == i)[0])
            class_indices = remaining_indices[class_indices]
            if len(class_indices) < k_shot:
                raise ValueError(f"Not enough samples in class {i} for k-shot learning")
            class_indices = np.random.choice(class_indices, k_shot, replace=False)
            train_indices.extend(class_indices)
    else:
        num_train = int(len(y) * k_shot)  # Note: now k_shot is ratio of total samples, not remaining ones
        if num_train > len(remaining_indices):
            raise ValueError("Not enough remaining samples for train set with the given k_shot ratio")
        train_indices = np.random.choice(remaining_indices, num_train, replace=False)

    train_mask = np.isin(np.arange(len(y)), train_indices)
    val_mask = np.isin(np.arange(len(y)), val_indices)
    test_mask = np.isin(np.arange(len(y)), test_indices)

    return train_mask, val_mask, test_mask


def fit_logistic_regression(X, y, k_shot):
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    X = normalize(X, norm='l2')

    accuracies = []

    train_mask, val_mask, test_mask = split_data(y, k_shot)

    X_train, y_train = X[train_mask], y_one_hot[train_mask]
    X_val, y_val = X[val_mask], y_one_hot[val_mask]
    X_test, y_test = X[test_mask], y_one_hot[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 11)
    #cv = ShuffleSplit(n_splits=5, test_size=0.5)
    cv = StratifiedKFold(n_splits=2)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                       n_jobs=5, cv=cv, verbose=0)

    #clf.fit(X_train, y_train)
    y_train_labels = np.argmax(y_train, axis=1)
    clf.fit(X_train, y_train_labels)

    y_pred = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

    test_acc = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(test_acc)
    return accuracies
