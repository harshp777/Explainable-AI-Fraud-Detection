import optuna
from pathlib import Path
import pickle


    #nested dictionary fir differene models to test

def get_hyperparameters(trial):



    return {


    'XGBClassifier':
{
    'objective': 'binary:logistic',
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.5, log=True),
    'max_depth': trial.suggest_int('max_depth', 1, 100),
    'subsample': trial.suggest_float('subsample', 0.5, 1),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
    'gamma': trial.suggest_float('gamma', 1e-8, 10, log=True),
    'alpha': trial.suggest_float('alpha', 1e-8, 10, log=True),
    'lambda': trial.suggest_float('lambda', 1e-8, 10, log=True),
}
,



    'RandomForestClassifier':
    {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),

    }





    }


"""
if __name__ == '__main__':

    curr_dir= Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    hyperparameters_path = Path(home_dir.as_posix()+ '/models/hyperparameters.pkl')
    pickle.dump(hyperparameters, open(hyperparameters_path, 'wb'))


"""
