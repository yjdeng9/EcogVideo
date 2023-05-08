import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap


def load_data():
    with open('ecog/full_X_subset_v2.pickle', 'rb') as f:
        data_list = pickle.load(f)

    x = np.stack(data_list)

    with open('ecog/full_Y_labels_v2.pickle', 'rb') as f:
        labels_list = pickle.load(f)

    label_names = [tup[0] for tup in labels_list]
    labels = [tup[1] for tup in labels_list]

    labels = np.array(labels)
    return x, labels, label_names


def train_trad(x,labels, seed=86):

    # Reshape the input data
    X = x.reshape((27, -1))
    y = labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # Create a k-NN classifier and fit it to the training data
    clf = KNeighborsClassifier(n_neighbors=2)
    # clf = RandomForestClassifier()
    # clf = XGBClassifier()
    clf.fit(X_train, y_train)
    clf.fit(X,y)

    # Use the classifier to predict the labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier's performance on the test data
    # train_accuracy = clf.score(X_train, y_train)
    # print(f"Train_accuracy: {train_accuracy:.2f}")
    train_accuracy = clf.score(X, y)
    print(f"Train_accuracy: {train_accuracy:.2f}")

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    return clf

def train_shap(x,labels):
    import xgboost
    # X, y = shap.datasets.boston()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    X = []
    for sub_x in x:
        sub_x = pca.fit_transform(sub_x)
        X.append(sub_x)
    X = np.array(X)
    X = X.reshape((27, -1))
    y = labels
    print(X.shape)

    import pandas as pd
    X = pd.DataFrame(X,columns=['Channel %d'%c for c in range(160)])

    label_names = ['Body','Face','Digit','Hira', 'Kanji', 'Line', 'Object']

    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(X.shape)
    print(shap_values.shape)

    unique_y = np.unique(y)

    for i in range(len(unique_y)):
        plt.figure(figsize=(5,3))
        label_name = label_names[unique_y[i]]
        plt.title(f"{label_name}", fontsize=25)
        idx = [ii for ii in range(len(y)) if y[ii] == unique_y[i]]
        plt.tight_layout()
        shap.summary_plot(shap_values[idx, :], X.iloc[idx, :])

    shap_interaction_values = explainer.shap_interaction_values(X)
    shap.summary_plot(shap_interaction_values, X)



def explian_in_train(final_data, final_labels):

    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(10, test_size=0.1)

    clf = XGBClassifier(n_setimators=100,seed = 86)
    # scores = cross_val_score(clf, final_data, final_labels, cv=cv)
    # print(scores)

    clf.fit(final_data, final_labels)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(final_data)
    shap.force_plot(explainer.expected_value, shap_values, final_data)

    shap_interaction_values = explainer.shap_interaction_values(final_data)
    shap.summary_plot(shap_interaction_values, final_data)

def main():
    # load data
    x, labels, label_names = load_data()
    print(x.shape)
    print(labels.shape)

    # for i in range(10):
    #     # 获取随机数
    #     seed = np.random.randint(0, 100)
    #     print(f"seed: {seed}")
    #     train_trad(x, labels,seed=seed)

    train_trad(x, labels, seed=86)

    # train_shap(x, labels)
    # train_trad(x, labels, seed=86)

    # # Reshape the input data
    # X = x.reshape((27, -1))
    # y = labels
    #
    # explian_in_train(X, y)
    #
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)




def model_compare():
    models = ['KNN','SVM', 'RandomForest','XGBoost']


if __name__ == '__main__':
    main()