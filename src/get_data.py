import numpy as np 
import pandas as pd 
from datasets import load_dataset
#  from scikit-learn import train_test_split

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
df = ds["train"].to_pandas()

print(df.info())
