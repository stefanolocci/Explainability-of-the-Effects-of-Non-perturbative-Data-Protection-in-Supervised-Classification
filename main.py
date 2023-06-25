import warnings
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from matplotlib import pyplot as plt
from lime import lime_tabular

warnings.filterwarnings("ignore")


def plot_histograms(data_list, bins, labels, title, xlabel, ylabel):
    """
    Plot histograms of multiple datasets.

    Args:
        data_list (list): List of data arrays to plot.
        bins (int): Number of bins in the histogram.
        labels (list): List of labels for each dataset.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    fig, ax = plt.subplots()
    for data, label in zip(data_list, labels):
        ax.hist(data, bins=bins, alpha=0.5, label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def load_dataset(path):
    """
    Load a dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path, sep=',').drop('ID', axis=1)


def one_hot_encoding(data, to_encode):
    """
    Perform one-hot encoding on selected columns of a DataFrame.

    Args:
        data (pandas.DataFrame): Input DataFrame.
        to_encode (list): List of columns to encode.
    """
    le = LabelEncoder()
    for col in to_encode:
        data[col] = data[col].astype('category')
        data[col] = le.fit_transform(data[col])


def split_X_Y(X, label):
    """
    Split the input DataFrame into feature matrix X and target variable Y.

    Args:
        X (pandas.DataFrame): Input DataFrame.
        label (str): Name of the target variable column.

    Returns:
        tuple: Tuple containing the feature matrix X and the target variable Y.
    """
    Y = X[label]
    X = X.drop(label, axis=1)
    return X, Y


def run_model(train, target, test, model_type):
    """
    Train and run a specified machine learning model.

    Args:
        train (pandas.DataFrame): Training data.
        target (pandas.Series): Target variable for training.
        test (pandas.DataFrame): Test data.
        model_type (str): Type of model to train and run.

    Returns:
        tuple: Tuple containing the trained model and the predicted labels for the test data.

    Raises:
        ValueError: If an invalid model type is specified.
    """
    if model_type == 'RF':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'SVM':
        svm_mod = svm.LinearSVC(dual=False)
        model = CalibratedClassifierCV(svm_mod, ensemble=False)
    elif model_type == 'XGB':
        model = xgb.XGBClassifier(objective="binary:hinge", random_state=42)
    else:
        raise ValueError('Invalid model type')
    model.fit(train, target)
    return model, model.predict(test)


def lime_explain(dataframe, test_feat, m, is_anon, exp, model_name):
    """
    Explains a single instance using LIME (Local Interpretable Model-Agnostic Explanations).

    Parameters:
        - dataframe (pd.DataFrame): The input dataset used for training the model.
        - test_feat (pd.DataFrame): The instance to be explained.
        - m: The trained model to be explained.
        - is_anon (bool): Indicates whether the explanation is for an anonymous instance.
        - exp (str): The name of the experiment.
        - model_name (str): The name of the model.

    """
    adult_samples = [20027]
    for idx in adult_samples:
        explainer = lime_tabular.LimeTabularExplainer(dataframe.to_numpy(), mode='classification',
                                                      feature_names=dataframe.columns)

        explanation = explainer.explain_instance(test_feat.loc[[idx]].to_numpy()[0], m.predict_proba,
                                                 num_features=len(dataframe.columns))
        explanation.as_pyplot_figure()
        plt.title('LIME features values')
        plt.tight_layout()
        plt.show()
        # produce explanations as an HTML file
        # html_data = explanation.as_html()
        # if is_anon:
        #     file_name = exp + '_explain_anon'
        # else:
        #     file_name = exp + '_explain_orig'
        # f = open(f"explanations/lime/adult/anon/{model_name}/{file_name}_1_test.html", mode='a')
        # f.write(html_data)
        # f.close()


def shap_explain(m, test_feat, k, mod_name, ldiv):
    """
    Explains a model using SHAP (SHapley Additive exPlanations).

    Parameters:
        - m: The trained model to be explained.
        - test_feat (pd.DataFrame): The test features used for generating explanations.
        - k (int): The number of features to display in the SHAP beeswarm plot.
        - mod_name (str): The name of the model.
        - ldiv: Parameter for SHAP beeswarm plot.

    """
    explainer = shap.Explainer(m.predict, test_feat)
    shap_values = explainer(test_feat)
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    path = 'explanations/shap/diabetes/anon/' + mod_name + '/shap_beeswarm_k' + str(k)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_features_analysis(data):
    """
    Plots the distribution of each feature in the dataset.

    Parameters:
        - data (pd.DataFrame): The dataset to analyze.

    """
    data = data.drop('ID', axis=1)
    for col in data.columns:
        unique_values = data[col].unique()
        for val in unique_values:
            occurrences = len(data[data[col] == val])
            plt.barh(val, occurrences)
        plt.xlabel('Occurrences')
        plt.ylabel('Feature values')
        plt.title('Distribution of ' + col)
        plt.show()


def cross_validation(alg, X_train, Y_train, measure, folds=10):
    """
    Performs cross-validation on a given algorithm.

    Parameters:
        - alg: The algorithm to evaluate.
        - X_train (pd.DataFrame): The training features.
        - Y_train: The training labels.
        - measure (str): The evaluation measure ('acc' for accuracy or 'f1' for F1 score).
        - folds (int): The number of cross-validation folds.

    Returns:
        - acc.mean() (float): The mean value of the evaluation measure across the folds.

    """
    kf = KFold(n_splits=folds, shuffle=True)
    acc = []
    i = 1
    for train_index, test_index in kf.split(X_train, Y_train):
        # print('{}-Fold'.format(i))
        fX_train, fX_test = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        fy_train, fy_test = Y_train[train_index], Y_train[test_index]
        alg.fit(fX_train, fy_train)
        fy_pred = alg.predict(fX_test)
        if measure == 'acc':
            curr = accuracy_score(fy_test, fy_pred, normalize=True)
        elif measure == 'f1':
            curr = f1_score(fy_test, fy_pred, average='micro')
        acc.append(curr)
        i = i + 1

    acc = pd.Series(acc)
    return acc.mean()


def plot_shap_rank():
    adult_shap_rank_KNN = [[1, 1, 1, 1, 5], [3, 3, 3, 2, 1], [4, 4, 4, 4, 3], [2, 2, 2, 3, 2], [5, 5, 5, 5, 4]]
    adult_shap_rank_RF = [[1, 1, 1, 2, 5], [4, 4, 4, 3, 2], [3, 2, 2, 1, 1], [2, 3, 3, 4, 3], [5, 5, 5, 5, 4]]
    adult_shap_rank_SVM = [[2, 1, 1, 1, 5], [1, 2, 3, 2, 1], [4, 4, 2, 4, 3], [3, 3, 4, 3, 2], [5, 5, 5, 5, 4]]
    adult_shap_rank_XGB = [[1, 1, 1, 3, 5], [4, 4, 2, 2, 1], [3, 3, 4, 1, 2], [2, 2, 3, 4, 3], [5, 5, 5, 5, 4]]
    diabetes_shap_rank_KNN = [[1, 3, 3, 3, 3], [2, 2, 2, 2, 2], [3, 1, 1, 1, 1], [4, 4, 4, 4, 4]]
    diabetes_shap_rank_RF = [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1], [4, 4, 4, 4, 4]]
    diabetes_shap_rank_SVM = [[2, 4, 4, 4, 4], [3, 2, 2, 2, 2], [1, 1, 1, 1, 1], [4, 3, 3, 3, 3]]
    diabetes_shap_rank_XGB = [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1], [4, 4, 4, 4, 4]]
    ph_l2_shap_rank_KNN = [[3, 5, 5, 5, 3], [6, 10, 10, 10, 8], [4, 4, 3, 8, 5], [9, 9, 7, 3, 4], [2, 3, 4, 2, 2],
                           [8, 8, 6, 7, 7], [1, 2, 1, 1, 1], [7, 7, 9, 4, 6], [3, 1, 1, 6, 9], [10, 6, 8, 9, 10]]
    ph_l2_shap_rank_RF = [[4, 4, 4, 5, 3], [9, 7, 10, 8, 9], [2, 5, 6, 4, 4], [10, 10, 7, 7, 8], [1, 3, 3, 3, 2],
                          [7, 9, 9, 9, 6], [3, 2, 1, 1, 1], [6, 6, 8, 6, 5], [5, 1, 2, 2, 7], [8, 8, 5, 10, 10]]
    ph_l2_shap_rank_SVM = [[10, 10, 10, 5, 8], [8, 9, 9, 4, 3], [2, 8, 8, 6, 6], [1, 7, 7, 3, 1], [5, 6, 6, 10, 9],
                           [9, 5, 5, 9, 5], [4, 4, 4, 2, 4], [6, 3, 3, 8, 7], [7, 2, 2, 7, 10], [3, 1, 1, 1, 2]]
    ph_l2_shap_rank_XGB = [[4, 4, 4, 5, 4], [6, 10, 10, 7, 9], [1, 5, 5, 3, 3], [9, 8, 6, 6, 7], [2, 2, 3, 2, 1],
                           [7, 9, 9, 9, 6], [5, 1, 1, 1, 2], [10, 6, 8, 8, 5], [3, 3, 2, 4, 8], [8, 7, 7, 10, 10]]
    ph_l3_shap_rank_KNN = [[3, 4, 5, 8, 3], [5, 10, 10, 10, 8], [4, 5, 4, 7, 5], [9, 7, 6, 4, 4], [2, 2, 3, 1, 2],
                           [8, 9, 9, 6, 7], [1, 3, 1, 2, 1], [7, 8, 8, 3, 6], [5, 1, 2, 5, 9], [10, 6, 7, 9, 10]]
    ph_l3_shap_rank_RF = [[4, 1, 4, 5, 3], [9, 10, 10, 9, 9], [2, 5, 6, 4, 4], [10, 9, 7, 7, 8], [1, 3, 3, 2, 2],
                          [7, 8, 8, 8, 6], [3, 4, 1, 1, 1], [6, 7, 9, 6, 5], [5, 2, 2, 3, 7], [8, 6, 5, 10, 10]]
    ph_l3_shap_rank_SVM = [[10, 10, 10, 5, 8], [8, 9, 9, 1, 3], [2, 8, 8, 8, 6], [1, 7, 7, 2, 1], [5, 6, 6, 7, 9],
                           [9, 5, 5, 10, 5], [4, 4, 4, 4, 4], [6, 3, 3, 9, 7], [7, 2, 2, 6, 10], [3, 1, 1, 3, 2]]
    ph_l3_shap_rank_XGB = [[4, 1, 4, 4, 4], [6, 10, 8, 7, 9], [1, 4, 5, 3, 3], [9, 9, 6, 5, 7], [2, 2, 1, 2, 1],
                           [7, 8, 9, 9, 6], [5, 5, 2, 1, 2], [10, 7, 10, 8, 5], [3, 3, 3, 6, 8], [8, 6, 7, 10, 10]]

    k_values = [1, 2, 20, 50, 100]
    adult_labels = ['age', 'education', 'marital-status', 'occupation', 'race']
    diabetes_labels = ['blood_glucose', 'age', 'HbA1c', 'hypertension']
    ph_labels = ['C1', 'S1', 'C2', 'S2', 'C3', 'S3', 'C4', 'S4', 'C5', 'S5']

    for i in range(len(diabetes_shap_rank_RF)):
        plt.plot(k_values, diabetes_shap_rank_RF[i], label=diabetes_labels[i])
    # plt.yticks([5, 4, 3, 2, 1])
    plt.yticks([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    plt.gca().invert_yaxis()
    plt.xticks([1, 2, 20, 50, 100])
    plt.xlabel('k-values')
    plt.ylabel('Rank')
    plt.title('KNN features rank as k varies')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=4)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_anon = True
    dataset = 'adult'
    f1_total_scores = []
    k_values = [2, 5, 10, 20, 50, 70, 100]
    # k_values = [100]
    models = []
    k = 1
    l = 1
    dataset_path = ''
    cat_cols = []
    label = ''
    test_feat_list = []
    model_names = ['RF', 'KNN', 'SVM', 'XGB']
    for m_name in model_names:
        for k in k_values:
            experiment = "Adult_k_{}".format(k)
            #     for l in [2, 3]:
            if run_anon:
                if dataset == 'adult':
                    dataset_path = 'anonymized_data/adult/mondrian/anon_ID_full/anonymized_' + str(k) + '.csv'
                    cat_cols = ['age', 'gender', 'race', 'marital-status', 'education', 'native-country',
                                'relationship', 'workclass', 'occupation', 'income']
                    label = 'income'
                elif dataset == 'poker_hand':
                    dataset_path = 'anonymized_data/poker_hand/mondrian_ldiv/anonypy/poker_hand_anon_k_' + str(
                        k) + '_l_' + str(
                        l) + '.csv'
                    cat_cols = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5']
                    label = 'Label'
                elif dataset == 'diabetes':
                    dataset_path = "anonymized_data/diabetes/mondrian/anonypy/diabetes_anon_k_" + str(k) + '.csv'
                    cat_cols = ['gender', 'smoking_history', 'age', 'hypertension', 'HbA1c_level',
                                'blood_glucose_level']
                    label = 'diabetes'
            else:
                if dataset == 'adult':
                    dataset_path = 'data/adult/adult_ID.csv'
                    cat_cols = ['gender', 'race', 'marital-status', 'education', 'native-country', 'relationship',
                                'workclass', 'occupation', 'income']
                    label = 'income'
                elif dataset == 'poker_hand':
                    dataset_path = 'data/poker_hand/poker_hand_train.csv'
                    cat_cols = []
                    label = 'Label'
                elif dataset == 'diabetes':
                    dataset_path = 'data/diabetes/diabetes_id_first.csv'
                    cat_cols = ['gender', 'smoking_history']
                    label = 'diabetes'

            X = load_dataset(dataset_path)

            if cat_cols:
                one_hot_encoding(X, cat_cols)
            X, Y = split_X_Y(X, label=label)

            train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.20,
                                                                                        random_state=42)
            model, predictions = run_model(train_features, train_labels, test_features, m_name)

            # PERFORMANCES
            # cv = KFold(n_splits=10, random_state=42, shuffle=True)
            # acc_score = cross_val_score(model, X.to_numpy(), Y, scoring='accuracy', cv=cv, n_jobs=-1)
            # f1_score = cross_val_score(model, X.to_numpy(), Y, scoring='f1_weighted', cv=cv, n_jobs=-1)

            # PERFORMANCE REPORT
            # print("\n##### " + m_name + " k = " + str(k) + " #####")
            # print('Accuracy: %.3f (%.3f)' % (mean(acc_score), std(acc_score)))
            # print('f1: %.3f (%.3f)' % (mean(f1_score), std(f1_score)))

            lime_explain(X, test_feat=test_features, m=model, is_anon=run_anon, exp=experiment, model_name=m_name)
            # shap_explain(model, test_feat=test_features, k=k, mod_name=m_name, ldiv=l)
