# !pip install -U scikit-learn
import lightning as L
from joblib import dump
from lightning.app.storage import Drive
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1234


class SKLearnTraining(L.LightningWork):
    def __init__(self):
        # we use CloudCompute API to configure the machine-related config
        # create a CPU machine with 10 GB disk size
        # https://lightning.ai/docs/stable/core_api/lightning_work/compute.html
        super().__init__(cloud_compute=L.CloudCompute("cpu", disk_size=10))

        # cloud persistable storage for model checkpoint
        self.model_storage = Drive("lit://checkpoints")

    def run(self):
        # Step 1
        # Download the dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split the dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=RANDOM_STATE
        )

        # Step 2
        # Intialize the model
        clf = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)

        # Step 3
        # Train the model
        clf = clf.fit(X_train, y_train)

        # check accuracy
        print(f"train accuracy: {clf.score(X_train, y_train)}")
        print(f"test accuracy: {clf.score(X_test, y_test)}")

        # Step 4
        # Save the model to disk
        dump(clf, "model.joblib")

        # Step 5
        # Put the model file to a persistable storage
        self.model_storage.put("model.joblib")
        print("model trained and saved successfully")


component = SKLearnTraining()
app = L.LightningApp(component)
