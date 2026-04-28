
from datasets import load_dataset

def retrieve_dataset(file):
    if file == "common":
        ds = load_dataset("gmanolache/CrypticBio", data_files="CrypticBio-Benchmark/CrypticBio-Common.csv", split="train")
    elif file == "full":
        ds = load_dataset("gmanolache/CrypticBio", split="train")

    return ds

if __name__ == "__main__":
    ds = retrieve_dataset(file="common")
    print(f"Dataset loaded, first entry:")
    print(ds[0])