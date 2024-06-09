from analysis.src_forwards.preprocessing.compose import Composer
from analysis.src_forwards.preprocessing.processor import (
    ApplyDataFrame,
    ConditionImputer,
    DropColumns,
    DropNa,
    FillnaImputer,
    GroupbyImputer,
)
from analysis.src_forwards.preprocessing.utils.assign_youth_player import youth_player

forwards_preprocessor = Composer(
    [
        ApplyDataFrame(features="is_youth_player", function=youth_player),
        ConditionImputer(
            features=["is_youth_player", "signing_fee_euro_mill"],
            condition="True",
            value=0,
        ),
        GroupbyImputer(
            features="signing_fee_euro_mill",
            groupby=["league", "age"],
            method="median",
        ),
        GroupbyImputer(
            features="market_value_euro_mill",
            groupby=["league", "age"],
            method="median",
        ),
        FillnaImputer(features="signed_from", value="Unknown"),
        FillnaImputer(features="signed_year", column_fill="season"),
        FillnaImputer(features="foot", method="mode"),
        ConditionImputer(
            features=["shots_on_target", "goals_per_shot_on_target"],
            condition=0,
            value=0,
        ),
        ConditionImputer(features=["shots", "avg_shot_distance"], condition=0, value=0),
        GroupbyImputer(
            features="avg_shot_distance", groupby="position", method="median"
        ),
        ConditionImputer(features=["shots", "goals_per_shot"], condition=0, value=0),
        ConditionImputer(
            features=["shots", "shots_on_target_pct"], condition=0, value=0
        ),
        ConditionImputer(
            features=["long_passes_attempted", "long_pass_completion_pct"],
            condition=0,
            value=0,
        ),
        ConditionImputer(
            features=["medium_passes_completed", "medium_pass_completion_pct"],
            condition=0,
            value=0,
        ),
        ConditionImputer(
            features=["short_passes_completed", "short_pass_completion_pct"],
            condition=0,
            value=0,
        ),
        ConditionImputer(
            features=["passes_attempted", "pass_completion_pct"], condition=0, value=0
        ),
        DropNa(),
        DropColumns(
            features=[
                "player",
                "country",
                "signed_from",
                "squad",
                "player_id",
                "annual_wages_euros",
                "age_range",
                "rk",
                "nation",
                "pos",
                "comp",
                "born",
                "general_pos",
                "is_youth_player",
            ]
        ),
    ]
)
