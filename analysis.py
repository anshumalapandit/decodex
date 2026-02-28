import pandas as pd
import os


# ---------------------------------------------------
# 1. LOAD TRADES FILE
# ---------------------------------------------------

def load_trades_file(file_path):
    df = pd.read_excel(file_path, engine="xlrd")

    df.columns = df.columns.str.strip().str.upper()

    print("COLUMNS FOUND:")
    print(df.columns.tolist())

    return df

# ---------------------------------------------------
# 2. STANDARDIZE REQUIRED COLUMNS
# ---------------------------------------------------

# def prepare_trades_dataframe(df):
    """
    Extract only required columns for structural analysis.
    """

    required_columns = [
        "BUY CLIENT CODE",
        "SELL CLIENT CODE",
        "TRADE_QUANTITY",
        "TRADE_RATE",
        "TRADE_TIME",
    ]

    # Normalize column names (replace underscores with space if needed)
    df.columns = df.columns.str.replace("_", " ")

    df = df[required_columns].copy()

    # Rename to consistent internal format
    df.rename(columns={
        "BUY CLIENT CODE": "buyer",
        "SELL CLIENT CODE": "seller",
        "TRADE_QUANTITY": "quantity",
        "TRADE_RATE": "price",
        "TRADE_TIME": "time"
    }, inplace=True)

    return df

def prepare_trades_dataframe(df):
    """
    Extract only required columns for structural analysis.
    """

    required_columns = [
        "BUY CLIENT CODE",
        "SELL CLIENT CODE",
        "TRADE_QUANTITY",
        "TRADE_RATE",
        "TRADE_TIME",
    ]

    # Make sure column names are stripped + uppercase (already done)
    df.columns = df.columns.str.strip().str.upper()

    # Select required columns
    df = df[required_columns].copy()

    # Rename to internal names
    df.rename(columns={
        "BUY CLIENT CODE": "buyer",
        "SELL CLIENT CODE": "seller",
        "TRADE_QUANTITY": "quantity",
        "TRADE_RATE": "price",
        "TRADE_TIME": "time"
    }, inplace=True)

    return df

# ---------------------------------------------------
# 3. CLIENT ACTIVITY SUMMARY
# ---------------------------------------------------

def compute_client_activity(df):
    """
    Compute total trades per client (buy + sell).
    """

    buy_counts = df.groupby("buyer").size()
    sell_counts = df.groupby("seller").size()

    total_activity = buy_counts.add(sell_counts, fill_value=0)

    activity_df = total_activity.reset_index()
    activity_df.columns = ["client", "total_trades"]

    return activity_df


# ---------------------------------------------------
# 4. COUNTERPARTY PAIR FREQUENCY
# ---------------------------------------------------

def compute_pair_frequency(df):
    """
    Count number of trades between each buyer-seller pair.
    """

    pair_counts = (
        df.groupby(["buyer", "seller"])
        .size()
        .reset_index(name="trade_count")
    )

    return pair_counts