import sys
import os
import pandas as pd
import numpy as np


def error_exit(msg):
    print(f"Error: {msg}")
    sys.exit(1)


def main():
    # 1. Check number of parameters
    if len(sys.argv) != 5:
        error_exit(
            "Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputResultFile>"
        )

    input_file = sys.argv[1]
    weights_str = sys.argv[2]
    impacts_str = sys.argv[3]
    output_file = sys.argv[4]

    # 2. File existence check
    if not os.path.isfile(input_file):
        error_exit("Input file not found")

    # 3. Read CSV
    try:
        df = pd.read_csv(input_file)
    except Exception:
        error_exit("Unable to read input file")

    # 4. Minimum columns check
    if df.shape[1] < 3:
        error_exit("Input file must contain three or more columns")

    # 5. Extract numeric data (2nd to last columns)
    data = df.iloc[:, 1:]

    # 6. Numeric validation
    try:
        data = data.astype(float)
    except ValueError:
        error_exit("From 2nd to last columns must contain numeric values only")

    # 7. Parse weights & impacts
    try:
        weights = [float(w) for w in weights_str.split(",")]
    except ValueError:
        error_exit("Weights must be numeric and comma separated")

    impacts = impacts_str.split(",")

    # 8. Length validation
    if not (len(weights) == len(impacts) == data.shape[1]):
        error_exit(
            "Number of weights, impacts and numeric columns must be the same"
        )

    # 9. Impact validation
    for imp in impacts:
        if imp not in ["+", "-"]:
            error_exit("Impacts must be either + or -")

    weights = np.array(weights)
    impacts = np.array(impacts)

    # ---------- TOPSIS IMPLEMENTATION ----------

    # Step 1: Normalize
    norm = np.sqrt((data ** 2).sum())
    normalized = data / norm

    # Step 2: Weighted normalized matrix
    weighted = normalized * weights

    # Step 3: Ideal best & worst
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(weighted.iloc[:, i].max())
            ideal_worst.append(weighted.iloc[:, i].min())
        else:
            ideal_best.append(weighted.iloc[:, i].min())
            ideal_worst.append(weighted.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Distances
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5: TOPSIS score
    score = dist_worst / (dist_best + dist_worst)

    # Step 6: Rank
    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    # 10. Save output
    try:
        df.to_csv(output_file, index=False)
    except Exception:
        error_exit("Unable to write output file")

    print("TOPSIS analysis completed successfully")


if __name__ == "__main__":
    main()
