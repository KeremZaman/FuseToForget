from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from fusion.utils.eval import get_dp, get_tpr_gap
from tqdm import tqdm
import numpy as np
import torch
import pickle
import os
from typing import Dict, List
import random
import warnings
import scipy
import evaluate
import argparse

"""Adapted from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/debias.py"""

# an abstract class for linear classifiers

class Classifier(object):

    def __init__(self):

        pass

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """

        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError


class SKlearnClassifier(Classifier):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis

    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)

    return P

def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P

def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=False, Y_train_main=None,
                             Y_dev_main=None, dropout_rate = 0) -> np.ndarray:
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :param dropout_rate: float, default: 0 (note: not recommended to be used with autoregressive=True)
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    I = np.eye(input_dim)

    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        clf = SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1./(1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale


        if by_class:
            #cls = np.random.choice(Y_train_main)  # uncomment for frequency-based sampling
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:

            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            # project

            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability and also generalize to the non-orthogonal case (e.g. with dropout),
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN) is roughly as accurate as this provided no dropout & regularization)
    """

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws

def encode_dataset(dataset_path: str, model_name: str, tokenizer: PreTrainedTokenizer, output: str):
    
    dataset = load_from_disk(dataset_path)
    train_data, test_data = dataset['train'], dataset['test']
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    X = []
    for sentence in tqdm(train_data['sentence']):
        inputs = tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt').to(device)
        enc_vec = model(**inputs).pooler_output.detach().cpu().numpy()[0, :]
        X.append(enc_vec)

    encoded_dataset_train = {'vectors': np.array(X), 'gender': np.array(train_data['gender']), 
                             'age': np.array(train_data['age']), 'label': np.array(train_data['label'])}

    X = []
    for sentence in tqdm(test_data['sentence']):
        inputs = tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt').to(device)
        enc_vec = model(**inputs).pooler_output.detach().cpu().numpy()[0, :]
        X.append(enc_vec)

    encoded_dataset_test = {'vectors': np.array(X), 'gender': np.array(test_data['gender']), 
                            'age': np.array(test_data['age']), 'label': np.array(test_data['label'])}
    
    encoded_dataset = {'train': encoded_dataset_train, 'test': encoded_dataset_test}

    with open(output, 'wb') as f:
        pickle.dump(encoded_dataset, f)

    return encoded_dataset


def run(dataset_path: str, model_name: str, tokenizer: str, encoding_output: str, guarded_attrs: List[str],
         num_classifiers: int = 200, is_autoregressive: bool = True, min_accuracy: float = 0.0):
    
    if not os.path.exists(encoding_output):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        encoded_dataset = encode_dataset(dataset_path, model_name, tokenizer, encoding_output)
    else:
        with open(encoding_output, 'rb') as f:
            encoded_dataset = pickle.load(f)

    X_train, Y_train = encoded_dataset['train']['vectors'], encoded_dataset['train']['label']
    X_test, Y_test = encoded_dataset['test']['vectors'], encoded_dataset['test']['label']
    Z_test_gender, Z_test_age = encoded_dataset['test']['gender'], encoded_dataset['test']['age']
    input_dim = X_train.shape[-1]

    #classifier_class = LogisticRegression
    
    classifier_class = SGDClassifier
    params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 64}

    #classifier_class = LinearSVC
    #params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}

    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)

    for guarded_attr in guarded_attrs:
        Z_train = encoded_dataset['train'][guarded_attr]
        P, rowspace_projections, Ws = get_debiasing_projection(classifier_class, params, num_classifiers,
                                                                input_dim, is_autoregressive,
                                                                  min_accuracy, X_train, Z_train, X_train,
                                                                    Z_train, by_class = False)
        
        X_train = P.dot(X_train.T).T

    clf = LogisticRegression(warm_start = True, penalty = 'l2',
                            solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                            verbose = 10, max_iter = 100, n_jobs = 64, random_state = 1)
    
    print(clf.fit(X_train, Y_train))
    

    preds = clf.predict(X_test)    
    dp_age = get_dp(preds, np.array(Z_test_age))
    dp_gender = get_dp(preds, np.array(Z_test_gender))

    tpr_gap_age = get_tpr_gap(Y_test, preds, Z_test_age)
    tpr_gap_gender = get_tpr_gap(Y_test, preds, Z_test_gender)

    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    accuracy = acc_metric.compute(predictions=preds, references=Y_test)['accuracy']
    f1 = f1_metric.compute(predictions=preds, references=Y_test)['f1']

    results = {'dp': {'age': dp_age, 'gender': dp_gender}, 'tpr_gap': {'age': tpr_gap_age, 'gender': tpr_gap_gender},
               'accuracy': accuracy, 'f1': f1}
    
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform INLP on given model')
    parser.add_argument('--dataset', help='', required=True)
    parser.add_argument('--model', help='', required=True)
    parser.add_argument('--tokenizer', help='', required=True)
    parser.add_argument('--encoding-output', help='', required=True)
    parser.add_argument('--guarded-attributes', nargs='+', choices=['gender', 'age'], help='', required=True)
    parser.add_argument('--num-classifiers', type=int, default=200, help='', required=False)
    parser.add_argument('--min-accuracy', type=float, default=0.0, help='', required=False)
    
    args = parser.parse_args()
    run(dataset_path=args.dataset, model_name=args.model, tokenizer=args.tokenizer, 
        encoding_output=args.encoding_output, guarded_attrs=args.guarded_attributes, 
        num_classifiers=args.num_classifiers, min_accuracy=args.min_accuracy)
