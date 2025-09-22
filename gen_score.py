# Run after 1.5K steps
# Just have to beat base model (mode="eval")

import json
import numpy as np
import os

def interpret_masking(min_idx, sorted_indices):
    """
    Interpret the masking strategy based on min_idx and sorted_indices.
    
    Args:
        min_idx: Index indicating which masking strategy was chosen
        sorted_indices: List of representation indices sorted from least to most surprising
    
    Returns:
        dict: Contains either 'isolate' or 'mask_out' key with representation names
    """
    representations = ["image", "grayscale", "semantic", "danger", "health", "proximity"]
    n = len(representations)  # 6
    
    if min_idx < n:
        # Isolate mode: keep only one representation, mask out everything else
        isolated_idx = sorted_indices[int(min_idx)]
        isolated_repr = representations[isolated_idx]
        return {"isolate": isolated_repr}
    elif min_idx < 2 * n:
        # Mask out mode: mask out specific representations based on descending order
        mask_out_count = int(min_idx) - n + 1  # How many to mask out
        
        # Get the most surprising representations (reverse order for masking)
        reversed_indices = sorted_indices[::-1]  # Most to least surprising
        
        # Mask out the top mask_out_count most surprising representations
        masked_indices = reversed_indices[:mask_out_count]
        masked_reprs = [representations[idx] for idx in masked_indices]
        
        return {"mask_out": masked_reprs}
    else:
        # min_idx == 12: "NA" mode - no masking
        return {"mode": "NA"}

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

def analyze_representations(input_file_path, output_file_path=None):
    """
    Analyze representation masking strategies and output to a new file.
    
    Args:
        input_file_path: Path to input jsonl file
        output_file_path: Path to output file (optional, will auto-generate if None)
    """
    if output_file_path is None:
        # Generate output filename based on input filename
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_representations.jsonl"
    
    with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            
            # Check if this line contains the required keys
            required_keys = ["step", "episode/min_idx"] + [f"episode/sorted_indices_{i}" for i in range(6)]
            
            if all(key in data for key in required_keys):
                step = int(data["step"])
                min_idx = int(data["episode/min_idx"])
                
                # Extract sorted indices
                sorted_indices = []
                for i in range(6):
                    sorted_indices.append(int(data[f"episode/sorted_indices_{i}"]))
                
                # Interpret the masking strategy
                masking_info = interpret_masking(min_idx, sorted_indices)
                
                # Create output entry
                output_entry = {"step": step}
                output_entry.update(masking_info)
                
                # Write to output file
                outfile.write(json.dumps(output_entry) + "\n")
    
    print(f"Representation analysis saved to: {output_file_path}")
    return output_file_path

def main():
    input_file = '/home/ei-lab/Documents/work/mk-dreamerv3/dreamerv3/dreamerv3/logdir/model-with-proximity/test.jsonl'
    
    # Generate scores as before
    mean, std = gen_score(input_file)
    print(f"Mean of episode/score: {mean:.4f}")
    print(f"Std of episode/score: {std:.4f}")
    
    # Analyze representations
    output_file = analyze_representations(input_file)
    
    # Show a few examples from the output
    print(f"\nFirst few representation analysis results:")
    with open(output_file, "r") as f:
        for i, line in enumerate(f):
            if i < 5:  # Show first 5 entries
                print(json.loads(line))
            else:
                break
    
if __name__ == '__main__':
    main()