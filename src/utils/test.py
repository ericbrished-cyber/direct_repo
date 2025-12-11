from src.utils.data_loader import DataLoader

DL = DataLoader()

print(DL.get_split_pmcids("FEW-SHOT"))
print(DL.get_few_shot_examples())