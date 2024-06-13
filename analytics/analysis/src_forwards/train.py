from rich import print

from analysis.src_forwards.config import ForwardsTrain as Settings
from analysis.src_forwards.training import ModelTrainer


def main() -> None:
    trainer = ModelTrainer(
        preprocessor=Settings.PREPROCESSOR,
        sklearn_pipeline=Settings.PIPELINE,
        models=Settings.MODELS,
        cv=Settings.CV,
        loader=Settings.LOADER,
    )
    trainer.run(
        input_path=Settings.INPUT,
        target_variable=Settings.TARGET,
    )

    print(trainer.cv_results_)
    print(trainer.train_scores_)
    print(trainer.test_scores_)


if __name__ == "__main__":
    main()
