import torch

# Paths
model_path = '/home/hice1/rpolaki3/scratch/rpolaki3/open_lth_data/lottery_fad1e89f2f31045fca3d2609b4b0355f/replicate_1/level_20/main/model_ep27_it0.pth'
mask_path = '/home/hice1/rpolaki3/scratch/rpolaki3/open_lth_data/lottery_fad1e89f2f31045fca3d2609b4b0355f/replicate_1/level_20/main/mask.pth'
log_file = 'sparsity_full_log.txt'

torch.set_printoptions(threshold=float('inf'))


# 1. Load — both are dicts of {layer_name: tensor}
model_sd = torch.load(model_path, map_location='cpu')
mask_sd  = torch.load(mask_path,  map_location='cpu')

def log_and_print(message, file):
    print(message)
    file.write(str(message) + '\n')

total_elements   = 0
non_zero_elements = 0

with open(log_file, 'w') as f:
    log_and_print('--- FULL TENSOR LOG ---', f)

    for layer_name in mask_sd:
        mask_tensor   = mask_sd[layer_name]
        weight_tensor = model_sd[layer_name]   # weights have same keys

        mask_binary  = (mask_tensor > 0).float()
        sparse_dense = weight_tensor * mask_binary

        total_elements    += sparse_dense.numel()
        non_zero_elements += torch.count_nonzero(sparse_dense).item()

        log_and_print(f'\n=== {layer_name} ===', f)
        log_and_print(f'Shape: {sparse_dense.shape}', f)

        log_and_print('\nFULL DENSE MASK:', f)
        log_and_print(mask_binary, f)

        log_and_print('\nFULL SPARSE DENSE MATRIX:', f)
        log_and_print(sparse_dense, f)

    # Overall stats at the end
    sparsity_percentage = ((total_elements - non_zero_elements) / total_elements) * 100
    log_and_print('\n' + '='*50, f)
    log_and_print(f'Total Elements:  {total_elements}',          f)
    log_and_print(f'Non-Zero Count:  {non_zero_elements}',       f)
    log_and_print(f'Sparsity:        {sparsity_percentage:.4f}%', f)
