from rich import print
from sklearn.model_selection import KFold

from analysis.gcp.loader import CSVLoader
from analysis.src_forwards.cross_validate import CrossValidate
from analysis.src_forwards.models import forwards_pipeline, regressors
from analysis.src_forwards.preprocessing.pipeline import forwards_preprocessor
from analysis.src_forwards.training._trainer import ModelTrainer


def main() -> None:
    cv = CrossValidate(
        metric="neg_mean_absolute_error",
        method=KFold(n_splits=5, shuffle=True, random_state=42),
    )
    models = regressors.models

    trainer = ModelTrainer(
        preprocessor=forwards_preprocessor,
        sklearn_pipeline=forwards_pipeline,
        models=models,
        cv=cv,
        loader=CSVLoader(),
    )
    trainer.run(
        input_path="wage_vals_stats/forwards.csv",
        target_variable="market_value_euro_mill",
    )

    print(trainer.cv_results_)
    print(trainer.train_scores_)
    print(trainer.test_scores_)


if __name__ == "__main__":
    main()
