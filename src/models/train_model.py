from sklearn.base import BaseEstimator
import pandas 
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import typing 
import logging 
import sklearn.exceptions 
import definitions

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename=definitions.LOGGING_DIRECTORY + "/model_train.py")


def fine_tune_model(
        k_cross: int,
        training_set: pandas.DataFrame, 
        target_variable: str,
        hyperparams: typing.Dict[str, typing.Union[list, str, int]],
        model: BaseEstimator,
        loss_function_or_scorer_metric: typing.Callable) -> BaseEstimator:

        """
        Function fine-tune ML Model using Cross-Validation with provided 
        hyperparameters and loss function
        
        Args:
            k_cross: K for cross-validation 
            training_set: (pandas.DataFrame) - training set
            target_variable: (str) - target variable we are trying to predict
            hyperparams: typing.Dict - dictionary, containing hyperparamaters for fine tuning 
            model: ML Model to tune 
            loss_function_or_scorer_metric - scoring function used for optimization
        """
        try:
            x_train = training_set.drop(columns=[target_variable])
            y_train = training_set[target_variable]

            cv_model = GridSearchCV(
                estimator=model,
                cv=StratifiedKFold(n_splits=k_cross),
                scoring=make_scorer(loss_function_or_scorer_metric),
                n_jobs=-1,
                param_grid=hyperparams
            )
            cv_model.fit(x_train, y_train)
            score_metric = cv_model.cv_results_['test_score'].mean()
            best_model = cv_model.best_estimator_
            return score_metric, best_model
            
        except(sklearn.exceptions.FitFailedWarning) as fit_err:
            logger.error(msg=fit_err)

        except Exception as err:
            logger.error(err)

        raise RuntimeError("""Failed to fine tune model, 
        check logs for more information""")
