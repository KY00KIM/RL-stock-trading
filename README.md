# RL Stock Trading

This repository contains an implementation of a reinforcement learning-based stock trading system. The system uses deep reinforcement learning algorithms to make trading decisions based on historical stock price data.

### Supported DRL-algorithms
- A2C : Advanced Actor-Critic
- DDPG : Deep Deterministic Policy Gradient

## Installation
### 1. Docker (Recommended)

```bash
# Build
docker build -t rl-stock-trading:0.1 .
docker run -it rl-stock-trading:0.1

```

### 2. Manual install

- Requirements
    - `poetry` : dependency manager ([link](https://python-poetry.org/))

   ```shell
   git clone https://github.com/KY00KIM/RL-stock-trading
   cd RL-stock-trading

   pip install poetry
   poetry install
   ```

## Run

```bash
# Download data & preprocess
poetry run main --mode setup &&\

# Train agents and save checkpoint
poetry run main --mode train --alg a2c &&\
poetry run main --mode train --alg ddpg &&\

# Backtesting
poetry run main --mode test
```

## Development
``` bash
.
├── poetry.lock     # Dependency Lock file
├── poetry.toml     # Poetry config
├── pyproject.toml  # Poetry project file
├── README.md       # README
├── Dockerfile      # Dockerfile
└── rl-stock-trading
    ├── agent.py    # DRL Agent implementation
    ├── config.py   # config for training/testing
    ├── env.py      # DRL Environment : stable-baseline&
    ├── __init__.py
    ├── __main__.py
    ├── run.py      # Entrypoint
    └── yf.py       # YahooFinance & Preprocessing utils
```

## Reference
- [OpenAI/baselines](https://github.com/openai/baselines)
- [DLR-RM/stable-baseline3](https://github.com/DLR-RM/stable-baselines3)
- [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL/)