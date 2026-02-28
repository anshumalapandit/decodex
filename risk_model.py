import pandas as pd


def compute_risk_score(concentration_df, reciprocity_ratio, loop_count):
    """
    Combine structural metrics into composite score.
    """

    # Normalize concentration
    concentration_df["norm_concentration"] = (
        concentration_df["counterparty_concentration"]
    )

    # Add global metrics to each client
    concentration_df["reciprocity"] = reciprocity_ratio
    concentration_df["loop_intensity"] = loop_count

    # Composite score (adjustable weights)
    concentration_df["risk_score"] = (
        0.4 * concentration_df["norm_concentration"]
        + 0.3 * concentration_df["reciprocity"]
        + 0.3 * (concentration_df["loop_intensity"] > 0).astype(int)
    )

    return concentration_df.sort_values("risk_score", ascending=False)