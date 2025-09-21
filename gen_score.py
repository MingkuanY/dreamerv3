# Run after 1.5K steps
# Just have to beat base model (mode="eval")

import json
import numpy as np

def gen_score(file_path):
    scores = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "episode/score" in data:
                scores.append(data["episode/score"])

    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    return mean, std

def main():
    mean, std = gen_score('/home/ei-lab/Documents/work/mk-dreamerv3/dreamerv3/dreamerv3/logdir/model-with-proximity/invert_health_surprise.jsonl')
    print(f"Mean of episode/score: {mean:.4f}")
    print(f"Std of episode/score: {std:.4f}")
    
if __name__ == '__main__':
  main()