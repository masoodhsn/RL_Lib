import argparse
from RL_Lib.agent import PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on a selected game")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment ID (default: CartPole-v1)")
    parser.add_argument("--steps", type=int, default=20000, help="Total training timesteps")
    args = parser.parse_args()

    agent = PPOAgent(env_id=args.env)
    agent.train(total_timesteps=args.steps)
    agent.evaluate()
    agent.save(f"ppo_{args.env}.zip")
