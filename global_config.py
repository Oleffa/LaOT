import os

dataset_path = os.getenv("LAOT_DATASET_PATH", "./data/nn_features/np/")
results_path = os.getenv("LAOT_RESULTS_PATH", "./models")

print(f"Base dataset path: {dataset_path}")
print(f"Base results path: {results_path}")


cfg = {
        'dataset_path' : dataset_path,
        'results_path' : results_path,
        }

available_methods = ['ae', 'vae']
available_experiments = ['features_ae']
