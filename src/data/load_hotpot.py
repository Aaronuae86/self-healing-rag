from datasets import load_dataset, load_dataset_builder


DATASET_NAME = "hotpot_qa"
CONFIG_NAME = "fullwiki"
SPLIT = "train"
SUBSET_SIZE = 3


def main():
    builder = load_dataset_builder(DATASET_NAME, CONFIG_NAME)
    split_names = list(builder.info.splits)
    print("Dataset split names:", split_names)

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=f"{SPLIT}[:{SUBSET_SIZE}]",
    )

    print("Loaded split names:", [SPLIT])
    print("Column names:", dataset.column_names)
    print("Number of loaded examples:", len(dataset))

    example = dataset[0]
    print("\nComplete example:")
    print(example)

    print("\nSelected fields and Python types:")
    for field in ("question", "answer", "context", "supporting_facts"):
        value = example[field]
        print(f"\n{field}:")
        print("Value:", value)
        print("Type:", type(value))


if __name__ == "__main__":
    main()
