from datasets import load_dataset, DatasetDict

def load_ticket_dataset() -> DatasetDict:
    
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")

    train_val_test = ds["train"].train_test_split(
        test_size = 0.2m seed=seed
    )
    val_test = ds["train"].train_test_split(
        test_size=0.5, seed=seed 
    )

    return DatasetDict({
        "train": train_val_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
