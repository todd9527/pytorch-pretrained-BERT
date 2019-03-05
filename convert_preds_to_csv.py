import argparse
import json

"""
example usage: convert_preds_to_csv.py --pred_file results/predictions.json --output_dir results
"""
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--pred_file", default=None, type=str, required=True)
parser.add_argument("--output_dir", default=None, type=str, required=True)
args = parser.parse_args()

with open(args.pred_file, "r") as f:
    data = json.load(f)
    with open(args.output_dir + "/dev_submission.csv", "w") as csv:
        csv.write("Id,Predicted\n")
        for hash in data:
            answer = data[hash].replace(",", "")
            csv.write(hash + "," + answer + "\n")

