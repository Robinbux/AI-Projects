import numpy as np

class Ensemble:

    @staticmethod
    def ensemble_predictions(*models_with_test_data):
        preds = []
        for model, test_data in models_with_test_data[0]:
            pred = model.predict(test_data)
            preds.append(pred)
        averaged_pred = np.average(preds, axis=0)
        return averaged_pred

if __name__=='__main__':
    from src_clean.enselmble.ensemble import Ensemble
    from joblib import dump, load
    # Load pretrained models
    lambda_mart_model = load('../lambda_mart.joblib')
    catboost_model = load('../catboost/catboost.joblib')
    bayesian_ridge_model = load('../bayesian_ridge/bayesian_ridge.joblib')

    model_data_list = [(lambda_mart_model, 'test'), (catboost_model, 'test'), (bayesian_ridge_model, 'test')]
    ensemble_preds = Ensemble.ensemble_predictions(model_data_list)