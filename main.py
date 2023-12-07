import rl_stock_trading
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train or test a machine learning model.")
    parser.add_argument("--mode", choices=["setup", "train", "test"], help="Select mode: train or test", required=True)
    parser.add_argument("--alg", choices=["a2c", "ddpg"], help="Select rl-algorithm: ddpg or a2c")

    args = parser.parse_args()

    if args.mode =="setup":
        rl_stock_trading.setup()
    elif args.mode == "train":
        rl_stock_trading.train_model(alg=args.alg)
    elif args.mode == "test":
        rl_stock_trading.test_model()

if __name__ == "__main__":
    main()