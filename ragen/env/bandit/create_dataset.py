"""
Preprocess dataset for bandit task
"""

import os
import json
from datasets import Dataset
import argparse
from ragen.env import BanditEnv

templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

intro = (
    "You are playing a multi-armed bandit game.\n"
    "\n"
    "Multi-Armed Bandit Quick Guide\n"
    "Goal: Maximize your total reward by choosing which arm to pull.\n"
    "\n"
    "Game Rules:\n"
    "1. There are {n_arms} slot machines (arms), numbered 1-{n_arms}\n"
    "2. Each machine has its own reward distribution\n"
    "3. Rewards are drawn from a normal distribution\n"
    "4. The mean reward for each machine is fixed but unknown\n"
    "\n"
    "Actions:\n"
    "Choose a number between 1 and {n_arms} to pull that arm\n"
    "Format: <answer> [number] </answer>\n"
    "\n"
    "Strategy Tips:\n"
    "- Balance exploration (trying different arms) and exploitation (using arms known to be good)\n"
    "- Keep track of the rewards you receive from each arm\n"
    "\n"
)

instruction_template = "{task_intro}\n[Current State]:\n{observation}\nChoose which arm to pull:"

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories for bandit environment.")
    parser.add_argument("--env", type=str, default="bandit", help="Environment name (default: 'bandit').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/bandit", help="Output directory (default: 'data/bandit').")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training instances (default: 10000).")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of test instances (default: 1000).")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])

    args = parser.parse_args()
    
    assert args.env == "bandit", f"Unsupported environment: {args.env}"
    os.makedirs(args.output, exist_ok=True)
    data_source = args.env

    n_arms = os.environ.get("N_ARMS")
    n_arms = int(n_arms)

    # Generate instructions
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    
    
    
    for seed in seeds:
        env = BanditEnv(n_arms=n_arms, seed=seed)
        observation = env.reset(seed=seed)
        instruction = instruction_template.format(
            task_intro=intro.format(n_arms=n_arms),
            observation=observation
        )
        instructions.append(instruction)

    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "rl",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }

    # Create datasets
    train_dataset = Dataset.from_list([
        _create_instance(args.seed + i, instructions[i]) 
        for i in range(args.train_size)
    ])
    
    test_dataset = Dataset.from_list([
        _create_instance(args.seed + i, instructions[i]) 
        for i in range(args.train_size, args.train_size + args.test_size)
    ])

    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Save datasets
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()