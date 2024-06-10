from rich import print

from analysis.gcp.loader import CSVLoader
from analysis.src_forwards.models import forwards_pipeline, regressors
from analysis.src_forwards.preprocessing.pipeline import forwards_preprocessor
from analysis.src_forwards.testing import ModelTester, Predict


def main() -> None:
    tester = ModelTester(
        preprocessor=forwards_preprocessor,
        sklearn_pipeline=forwards_pipeline,
        models=regressors.models,
        tester=Predict(),
        loader=CSVLoader(),
    )

    tester.run(
        input_path="wage_vals_stats/forwards.csv",
        target_variable="market_value_euro_mill",
    )

    print(tester.pred_results_)
    print(tester.performance_)

    # upload results to GCP
    # call results from GCP via API


if __name__ == "__main__":
    main()
