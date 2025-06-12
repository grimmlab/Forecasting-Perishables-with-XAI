import pandas as pd
from sklearn.model_selection import ShuffleSplit
import numpy as np
import optuna
import sklearn
import datetime
import xgboost
import csv
import shap
from sklearn.model_selection import train_test_split
import copy
import pickle
import matplotlib.pyplot as plt

dataset = "_no_calendar_year"

data = pd.read_csv("Joghurt_20-23" + dataset + ".csv")

data.replace(" ", np.nan, inplace=True)

n_splits = 5
model_name = "XGBoost"
# models: ard, bayesridge, elasticnet, lasso, ridge, xgboost


def get_one_hot_encoded_df(df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """
    Function delivering dataframe with specified columns one hot encoded

    :param df: dataset to use for encoding
    :param columns_to_encode: columns to encode

    :return: dataset with encoded columns
    """
    return pd.get_dummies(df, columns=columns_to_encode)


# one hot encode the data
print("-one-hot-encoding the data-")
if "_no_calendar" not in dataset:
    data = get_one_hot_encoded_df(
        df=data, columns_to_encode=["öffentlicherFeiertag", "Schulferien"]
    )
print("-one-hot-encoded the data-")
data = data.drop("Produkt", axis=1)


def drop_columns(df: pd.DataFrame, columns: list):
    """
    Function dropping all columns specified

    :param df: dataset used for dropping
    :param columns: columns which should be dropped
    """
    df.drop(columns=columns, inplace=True)


def encode_cyclical_features(df: pd.DataFrame, columns: list):
    """
    Function that encodes the cyclic features to sinus and cosinus distribution

    :param df: DataFrame to use for imputation
    :param columns: columns that should be encoded
    """
    for col in columns:
        max_val = df[col].max()
        df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        drop_columns(df=df, columns=col)


if "_no_calendar" not in dataset:
    encode_cyclical_features(df=data, columns=["Woche"])
data['MittlererPreisSonderangebote'] = data['MittlererPreisSonderangebote'].astype(float)

def get_indexes(df: pd.DataFrame, n_splits: int = n_splits):
    """
    Get the indexes for cv

    :param df: data that should be splited
    :param n_splits: number of splits for the cv
    :param datasplit: splitting method

    :return: train and test indexes
    """
    train_indexes = []
    test_indexes = []
    splitter = ShuffleSplit(n_splits=n_splits, random_state=69)
    for train_index, test_index in splitter.split(df):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes


def retrain(retrain: pd.DataFrame, model):
    """
    Implementation of the retraining for models with sklearn-like API.
    See :obj:`~ForeTiSHortiCo-Hortico.model._base_model.BaseModel` for more information.
    """
    x_train = retrain.drop("GesamtmengeVj", axis=1)
    y_train = retrain["GesamtmengeVj"]
    model.fit(x_train, y_train)


def predict(X_in: pd.DataFrame, model) -> np.array:
    """
    Implementation of a prediction based on input features for models with sklearn-like API.
    See :obj:`~ForeTiSHortiCo-Hortico.model._base_model.BaseModel` for more information.
    """
    X_in = X_in.drop("GesamtmengeVj", axis=1)
    prediction = model.predict(X_in)
    return prediction.flatten().round().astype(int)


def train_val_loop(train: pd.DataFrame, val: pd.DataFrame, model) -> np.array:
    """
    Implementation of a train and validation loop for models with sklearn-like API.
    See :obj:`~ForeTiSHortiCo-Hortico.model._base_model.BaseModel` for more information.
    """
    # train model
    retrain(train, model)
    # validate model
    return predict(X_in=val, model=model)

if dataset != "_no_calendar_year":
    test = data[data["Jahr"] == 2023]
    if dataset == "_no_calendar_year_yearly":
        data = data.drop("Jahr", axis=1)
        test = test.drop("Jahr", axis=1)
else:
    train_val, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

def objective(trial: optuna.trial.Trial):
    """
    Objective function for optuna optimization that returns a score

    :param trial: trial of optuna for optimization

    :return: score of the current hyperparameter config
    """

    train_indexes, val_indexes = get_indexes(df=train_val)

    max_depth = trial.suggest_int("max_depth", 2, 1000, step=10)
    n_estimators = trial.suggest_int("n_estimators", 500, 1000, step=50)
    gamma = trial.suggest_int("gamma", 0, 1000, step=10)
    reg_lambda = trial.suggest_float("reg_lambda", 0, 1000, step=1)
    reg_alpha = trial.suggest_float("reg_alpha", 0, 1000, step=1)
    learning_rate = trial.suggest_float("learning_rate", 0.025, 0.3, step=0.025)
    subsample = trial.suggest_float("subsample", 0.05, 1.0, step=0.05)
    colsample_bytree = trial.suggest_float(
        "colsample_bytree", 0.005, 1.0, step=0.005
    )

    # load the unfitted model to prevent information leak between folds
    unfitted_model = xgboost.XGBRegressor(
        random_state=42,
        verbosity=0,
        objective="reg:squarederror",
        tree_method="auto",
        max_depth=max_depth,
        n_estimators=n_estimators,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )

    objective_values = []

    for fold in range(n_splits):
        model = copy.deepcopy(unfitted_model)

        train, val = (
            train_val.iloc[train_indexes[fold]],
            train_val.iloc[val_indexes[fold]],
        )

        # run train and validation loop for this fold
        y_pred = train_val_loop(train=train, val=val, model=model)

        objective_value = sklearn.metrics.r2_score(
            y_true=val["GesamtmengeVj"], y_pred=y_pred
        )

        # store results
        objective_values.append(objective_value)

    current_val_result = float(np.mean(objective_values))

    return current_val_result


def create_new_study() -> optuna.study.Study:
    """
    Create a new optuna study.

    :return: a new optuna study instance
    """
    study_name = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + "-MODEL"
        + model_name
        + "-TRIALS"
        + str(200)
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20),
        load_if_exists=True,
    )

    return study


def run_optuna_optimization() -> dict:
    """
    Run whole optuna optimization for one model, dataset and datasplit.

    :return: dictionary with results overview
    """
    # Create a new study
    study = create_new_study()
    # Start optimization run
    study.optimize(lambda trial: objective(trial=trial), n_trials=200)
    print(study.best_trial.value)
    print(study.best_params)

    return study.best_params


best_params = run_optuna_optimization()

best_model = xgboost.XGBRegressor(**best_params)

retrain(train_val, best_model)

pickle.dump(best_model, open("models/" + dataset + "_xgb", 'wb'))

predictions = predict(X_in=test, model=best_model)

explainer = shap.Explainer(best_model.predict, test.drop("GesamtmengeVj", axis=1))
shap_values = explainer(test.drop("GesamtmengeVj", axis=1))

filename_expl = 'explainer' + dataset + '.sav'
pickle.dump(explainer, open("explainer/" + filename_expl, 'wb'))

filename = 'shapvalues' + dataset + '.sav'
pickle.dump(shap_values, open("shapvalues/" + filename, 'wb'))

for feature in ["kcalje100gr", "Zuckergehaltje100gr", "Fettgehalt_neu", "Bio", "Proteingehaltje100gr"]:
    shap.partial_dependence_plot(
        feature,
        best_model.predict,
        test.drop("GesamtmengeVj", axis=1),
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
        show=False
    )
    f = plt.gcf()
    f.savefig("partial_dependence_plots/shap.partial_dependence_plot" + dataset + feature + ".pdf", format='pdf', bbox_inches='tight')

np.savetxt("predictions/predictions" + dataset + "_xgb.csv", predictions, delimiter=",")
test.to_csv("testsets/test" + dataset + "_xgb.csv", index=False,
            sep=',', decimal='.', float_format='%.10f')
with open('best_params/best_params' + dataset + '_xgb.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(best_params.items())
with open('R2_scores/R2_score' + dataset + '_xgb.txt', 'w') as f:
    f.write("R2 Score: %.2f" % sklearn.metrics.r2_score(y_true=test["GesamtmengeVj"], y_pred=predictions))

def get_feature_importance(model) -> pd.DataFrame:
    """
    Get feature importances for models that possess such a feature, e.g. XGBoost

    :param model: model to analyze

    :return: DataFrame with feature importance information
    """
    feat_import_df = pd.DataFrame()
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1]
    feat_import_df["feature"] = data.drop(["GesamtmengeVj"], axis=1).columns[sorted_idx]
    feat_import_df["feature_importance"] = feature_importances[sorted_idx]

    return feat_import_df

feat_import_df = get_feature_importance(model=best_model)
feat_import_df.to_csv(
    "final_model_feature_importances/final_model_feature_importances" + dataset + "_xgb.csv",
    sep=",",
    decimal=".",
    float_format="%.10f",
    index=False,
)
