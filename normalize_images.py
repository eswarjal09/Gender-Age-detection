from torch.utils.data import DataLoader

dataloader = DataLoader(*torch_dataset*, batch_size=4096, shuffle=False, num_workers=4)
pop_mean = []
pop_std0 = []
pop_std1 = []
i = 0
for img, age, gender in train_loader:
    # shape (batch_size, 3, height, width)
    numpy_image = img.numpy()
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)
    i = i+1

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
pop_std1 = np.array(pop_std1).mean(axis=0)