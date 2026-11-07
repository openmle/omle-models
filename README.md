# OMLE Models

A notebook collection of classical machine learning models in OMLE format.
Each notebook contains the complete workflow: load data, train a pipeline,
evaluate it, export to OMLE, inspect its interactive graph, and compare native and
OMLE runtime predictions.

## Run locally

Use Python 3.10 or newer. From this repository:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook and run all cells from a fresh kernel. Select the environment
where the requirements were installed. No installation of this repository is needed.
The embedded OMLE viewer is interactive in a running notebook; GitHub previews
do not execute its JavaScript.

The shared requirements constrain pandas to 2.x for compatibility with OMLE
mixed-column inference. Restart the kernel after changing dependencies.

## Additional framework dependencies

The base requirements cover the scikit-learn notebooks. For the other examples,
install the matching requirements file in the same environment as your kernel:

```sh
# XGBoost and LightGBM
python -m pip install -r requirements-boosting.txt

# Spark ML (PySpark 3.5 or 4.x)
python -m pip install -r requirements-spark.txt
```

The Spark notebooks use Java 17 with `JAVA_HOME` configured and a local Spark
session; each aligns Python workers with the kernel interpreter and stops Spark
at the end. See the [PySpark installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).
All four Spark notebooks have passed end-to-end checks with **PySpark 3.5.9 and
4.3.0**, Python 3.11, and Java 17. Checks include export, structural validation,
viewer generation, native/OMLE prediction parity, and missing/unseen-category
probes for the mixed-type recipes. PySpark 4.3 requires Python 3.11 or newer.
Notebook outputs are cleared for Git; run the cells to generate results locally.
Training results can vary between Spark versions; each export is compared with
its own native fitted model.

On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`); see the
[LightGBM installation instructions](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst).

## Examples

| Task | Dataset | Framework | Model | Notebook |
| --- | --- | --- | --- | --- |
| Binary classification | Breast Cancer Wisconsin | scikit-learn | StandardScaler + LogisticRegression | [Open notebook](notebooks/classification/breast-cancer/sklearn-logistic-regression.ipynb) |
| Binary classification | Breast Cancer Wisconsin | XGBoost | XGBClassifier | [Open notebook](notebooks/classification/breast-cancer/xgboost-classifier.ipynb) |
| Binary classification | Breast Cancer Wisconsin | LightGBM | LGBMClassifier | [Open notebook](notebooks/classification/breast-cancer/lightgbm-classifier.ipynb) |
| Binary classification | Breast Cancer Wisconsin | Spark ML | StandardScaler + LogisticRegression | [Open notebook](notebooks/classification/breast-cancer/spark-logistic-regression.ipynb) |
| Multiclass classification | Iris | scikit-learn | RandomForestClassifier | [Open notebook](notebooks/classification/iris/sklearn-random-forest.ipynb) |
| Multiclass classification | Wine | scikit-learn | StandardScaler + LogisticRegression | [Open notebook](notebooks/classification/wine/sklearn-logistic-regression.ipynb) |
| Regression | Diabetes | scikit-learn | StandardScaler + Ridge | [Open notebook](notebooks/regression/diabetes/sklearn-ridge.ipynb) |
| Regression | Diabetes | scikit-learn | RandomForestRegressor | [Open notebook](notebooks/regression/diabetes/sklearn-random-forest.ipynb) |
| Clustering | Synthetic blobs | scikit-learn | StandardScaler + KMeans | [Open notebook](notebooks/clustering/blobs/sklearn-kmeans.ipynb) |
| Binary classification (mixed types) | Titanic | scikit-learn | ColumnTransformer + LogisticRegression | [Open notebook](notebooks/classification/titanic/sklearn-mixed-logistic-regression.ipynb) |
| Binary classification (mixed types) | Adult Census Income | scikit-learn | ColumnTransformer + LogisticRegression | [Open notebook](notebooks/classification/adult/sklearn-mixed-logistic-regression.ipynb) |
| Regression (mixed types) | Ames Housing | scikit-learn | ColumnTransformer + Ridge | [Open notebook](notebooks/regression/ames-housing/sklearn-mixed-ridge.ipynb) |
| Binary classification (mixed types) | Titanic | Spark ML | Imputer + StringIndexer + OneHotEncoder + LogisticRegression | [Open notebook](notebooks/classification/titanic/spark-mixed-logistic-regression.ipynb) |
| Binary classification (mixed types) | Adult Census Income | Spark ML | Imputer + StringIndexer + OneHotEncoder + RandomForestClassifier | [Open notebook](notebooks/classification/adult/spark-mixed-random-forest.ipynb) |
| Regression (mixed types) | Ames Housing | Spark ML | Imputer + StringIndexer + OneHotEncoder + LinearRegression | [Open notebook](notebooks/regression/ames-housing/spark-mixed-linear-regression.ipynb) |

## Data and artifacts

Datasets are loaded through their library or original public source; no dataset
copies are stored here. Each notebook documents attribution and licensing.
External downloads should use stable versions and checksums, with caches excluded
from Git.

Titanic, Adult, and Ames Housing use pinned OpenML dataset IDs and download into
`.cache/openml/`. The first run needs internet; later runs reuse the cache.
Notebooks record source metadata, checksums, selected columns, and column types.
Dataset-specific preprocessing and missing/unseen-category policies are explained
in each notebook.

Notebooks locate the repository root from the working directory and write to
`models/<task>/<dataset>/`. New recipes use `<recipe>-environment.json` so their
run records do not overwrite each other. Generated models and environment reports are ignored
by default. Small reviewed OMLE examples may be committed explicitly; complete
collections can be distributed through GitHub Releases. Native framework model
files are not required.

Seeds and exact package versions are recorded by the example. This initial
requirements file is not a lockfile; dependency versions can be pinned after
reviewing the first example in the intended environment.

## Verification

Each notebook displays its held-out quality metrics and requires OMLE predictions
to match its own native fitted model. Classification checks labels and probability
columns; regression checks numeric predictions; clustering checks cluster IDs
from the same fitted model. Structural OMLE validation and operator inspection
precede the interactive viewer. All notebook outputs are cleared for Git; execute the cells to display results
and the interactive viewer.

The random-forest recipes use float32 feature inputs to match scikit-learn tree
inference at split thresholds. Other recipes document their own preprocessing.
Do not compare independently trained frameworks for identical predictions.
