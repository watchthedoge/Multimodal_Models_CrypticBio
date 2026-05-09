from datasets import load_dataset
import numpy as np
from collections import Counter
def retrieve_dataset(file):
    if file == "common":
        ds = load_dataset("gmanolache/CrypticBio", data_files="CrypticBio-Benchmark/CrypticBio-Common.csv", split="train")
    elif file == "common_unseen":
        ds = load_dataset("gmanolache/CrypticBio", data_files="CrypticBio-Benchmark/CrypticBio-CommonUnseen.csv", split="train")
    elif file == "endangered":
        ds = load_dataset("gmanolache/CrypticBio", data_files="CrypticBio-Benchmark/CrypticBio-Endagered.csv", split="train")
    elif file == "full":
        ds = load_dataset("gmanolache/CrypticBio", split="train")
    else:
        raise ValueError(f"Unknown benchmark '{file}'. Expected one of: common, common_unseen, endangered, full.")

    return ds

def dataset_overview(ds):
    print("\n=== DATASET OVERVIEW ===")
    print(ds)

    print("\nColumns:")
    print(ds.column_names)

    print(f"\nNumber of samples: {len(ds)}")

    # Loop through columns and analyze distributions
    for col in ds.column_names:
        print(f"\n--- Column: {col} ---")
        values = ds[col]

        # Try categorical distribution
        try:
            counts = Counter(values)
            most_common = counts.most_common(10)
            print("Top values:")
            for val, count in most_common:
                print(f"{val}: {count}")
        except Exception:
            pass

        # Try numeric summary
        try:
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            if len(numeric_vals) > 0:
                print("Numeric stats:")
                print(f"  mean: {np.mean(numeric_vals):.4f}")
                print(f"  std: {np.std(numeric_vals):.4f}")
                print(f"  min: {np.min(numeric_vals)}")
                print(f"  max: {np.max(numeric_vals)}")
        except Exception:
            pass


if __name__ == "__main__":
    ds = retrieve_dataset(file="common")

    print("Dataset loaded, first entry:")
    print(ds[0])

    dataset_overview(ds)
