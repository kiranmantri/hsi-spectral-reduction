import pandas as pd
import sklearn.metrics

# from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, dataset_name: str, dimensions: int):
        self.hparams = {}
        self.hparams["dataset_name"] = dataset_name
        self.hparams["dimensions"] = dimensions
        self.hparams["is_trained"] = False

        # self.model = SGDClassifier(penalty='l1')
        # self.model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        self.model = XGBClassifier(n_estimators=10, n_jobs=8, max_depth=10, random_state=42)
        # self.model = XGBClassifier(n_estimators=2, n_jobs=8, max_depth=2, random_state=42)
        # self.model = SVC()
        # self.model = RidgeClassifier()

    @property
    def dataset_name(self):
        return self.hparams["dataset_name"]

    @property
    def dimensions(self):
        return self.hparams["dimensions"]

    @property
    def is_trained(self):
        return self.hparams["is_trained"]

    def train(self, data, targets):
        self.model.fit(X=data, y=targets)
        return self.model

    def evaluate(self, data, targets, labels: dict):
        _ = labels.pop("undefined", None)
        predicted = self.model.predict(data).ravel()

        confusion_matrix = sklearn.metrics.confusion_matrix(targets, predicted, labels=list(labels.values()))
        confusion_matrix = pd.DataFrame(confusion_matrix, columns=list(labels.keys()))
        confusion_matrix.index = confusion_matrix.columns

        confusion_matrix.apply(func=lambda item: item / item.sum(), axis=1)
        confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap='tab10')
        # plt.show()

        return predicted, confusion_matrix

    def predict(self, data):
        predicted = self.model.predict(data).ravel()
        return predicted
