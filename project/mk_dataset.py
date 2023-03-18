from datasets import load_dataset

dataset = load_dataset('audiofolder', data_dir='./dataset')
dataset.push_to_hub("JBJoyce/DENTAL_CLICK")



