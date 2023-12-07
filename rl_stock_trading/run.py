from .yf import yf_fetch_data, clean_data, data_split
from .config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TICKERS, INITIAL_AMOUNT
import argparse

def setup():
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    df = yf_fetch_data(TRAIN_START_DATE, TEST_END_DATE, TICKERS)
    df.sort_values(['date','tic'])
    df.shape, len(df.tic.unique()), df.head(), df.tail()
    processed = clean_data(df)
    processed = processed.ffill().bfill()
    print("Ticks:",processed["tic"].unique())
    train_data = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed, TEST_START_DATE, TEST_END_DATE)
    train_data.to_csv('./data/train_data.csv')
    test_data.to_csv('./data/test_data.csv')
    print("Setup complete.")

def train_model(alg: str):
    from .env import StockTradingEnv
    from .agent import DRLAgent
    import pandas as pd
    import time
    train_data = pd.read_csv('./data/train_data.csv')

    train_data = train_data.set_index(train_data.columns[0])
    train_data.index.names = ['']

    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension
    num_stock_shares = [0] * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": INITIAL_AMOUNT, 
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, 
        "sell_cost_pct": sell_cost_list, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4, 
    }
    state_space, stock_dimension

    train_env = StockTradingEnv(df = train_data, **env_kwargs)
    train_env_sb, _ = train_env.get_sb_env()

    if alg == "a2c":
        a2c = True
        ddpg = False
    elif alg == "ddpg":
        a2c = False
        ddpg = True
    else:
        assert False, "Invalid algorithm."

    agent = DRLAgent(env = train_env_sb)
    model_a2c = agent.get_model("a2c")
    model_ddpg = agent.get_model("ddpg")

    begin_time = time.time()
    trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000) if a2c else None
    trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=50000) if ddpg else None
    end_time = time.time()

    trained_a2c.save("checkpoint/agent_a2c") if a2c else None
    trained_ddpg.save("checkpoint/agent_ddpg") if ddpg else None
    print(f"Training complete. {alg} Agent : {end_time - begin_time} seconds")


def test_model():
    import pandas as pd
    from stable_baselines3 import A2C
    from stable_baselines3 import DDPG
    from .env import StockTradingEnv
    from .agent import DRLAgent
    import matplotlib.pyplot as plt
    import os

    a2c = os.path.exists("checkpoint/agent_a2c.zip")
    ddpg = os.path.exists("checkpoint/agent_ddpg.zip")
    if not os.path.exists("output"):
        os.makedirs("output")
    
    test_data = pd.read_csv('data/test_data.csv')
    test_data = test_data.set_index(test_data.columns[0])
    test_data.index.names = ['']

    trained_a2c = A2C.load("checkpoint/agent_a2c") if a2c else None
    trained_ddpg = DDPG.load("checkpoint/agent_ddpg") if ddpg else None

    stock_dimension = len(test_data.tic.unique())
    state_space = 1 + 2 * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    # test_env = StockTradingEnv(df = test_data, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
    test_env = StockTradingEnv(df = test_data, **env_kwargs)

    df_account_value_a2c, _ = DRLAgent.DRL_prediction(
        model=trained_a2c, 
        environment = test_env) if a2c else (None, None)
    df_account_value_ddpg, _ = DRLAgent.DRL_prediction(
        model=trained_ddpg, 
        environment = test_env) if a2c else (None, None)

    df_result_a2c = (
        df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
        if a2c
        else None
    )
    df_result_ddpg = (
        df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
        if ddpg
        else None
    )

    kospi = yf_fetch_data(start_date=TEST_START_DATE, end_date=TEST_END_DATE, ticker_list=["^KS11"])
    kospi = kospi[["date", "close"]]
    fst_day = kospi["close"][0]
    kospi = pd.merge(
        kospi["date"],
        kospi["close"].div(fst_day).mul(INITIAL_AMOUNT),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")
    result = pd.DataFrame(
        {
            "a2c": df_result_a2c["account_value"] if a2c else None,
            "ddpg": df_result_ddpg["account_value"] if ddpg else None,
            "dji": kospi["close"],
        }
    )
    # Jupyter / I-Python environment
    # %matplotlib inline
    plt.rcParams["figure.figsize"] = (15,5)
    plt.figure()
    result.plot()
    result.to_csv("output/result.csv")

def run():
    parser = argparse.ArgumentParser(description="Train or test a machine learning model.")
    parser.add_argument("--mode", choices=["setup", "train", "test"], help="Select mode: train or test", required=True)
    parser.add_argument("--alg", choices=["a2c", "ddpg"], help="Select rl-algorithm: ddpg or a2c")

    args = parser.parse_args()

    if args.mode =="setup":
        setup()
    elif args.mode == "train":
        train_model(alg=args.alg)
    elif args.mode == "test":
        test_model()

if __name__ == "__main__":
    run()
