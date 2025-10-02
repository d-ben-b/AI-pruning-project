import os
import pickle
import torch
from torchvision import datasets, transforms


def build_loaders(args):
    cache_dir = os.path.join(args.out, "index_cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_samples = os.path.join(cache_dir, "train_samples.pkl")
    val_cache_samples = os.path.join(cache_dir, "val_samples.pkl")
    train_cache_full = os.path.join(cache_dir, "train_dataset.pt")
    val_cache_full = os.path.join(cache_dir, "val_dataset.pt")

    def load_or_build(split, transform, cache_file_samples, cache_file_full):
        if getattr(args, "cache_mode", "full") == "samples":
            # ========= SAMPLES MODE =========
            if os.path.exists(cache_file_samples):
                print(f"✅ Loading {split} samples from {cache_file_samples}")
                with open(cache_file_samples, "rb") as f:
                    samples, class_to_idx = pickle.load(f)
                dataset = datasets.ImageFolder(
                    os.path.join(args.data_path, split), transform=transform
                )
                dataset.samples = samples
                dataset.imgs = samples
                dataset.class_to_idx = class_to_idx
            else:
                # ========= FULL MODE =========
                print(f"🔄 Building {split} samples (slow, first time)...")
                dataset = datasets.ImageFolder(
                    os.path.join(args.data_path, split), transform=transform
                )
                with open(cache_file_samples, "wb") as f:
                    pickle.dump((dataset.samples, dataset.class_to_idx), f)
                print(f"💾 Cached {split} samples to {cache_file_samples}")

        else:
            if os.path.exists(cache_file_full):
                print(f"✅ Loading FULL {split} dataset from {cache_file_full}")
                dataset = torch.load(cache_file_full, weights_only=False)
            else:
                print(f"🔄 Building FULL {split} dataset (slow, first time)...")
                dataset = datasets.ImageFolder(
                    os.path.join(args.data_path, split), transform=transform
                )
                torch.save(dataset, cache_file_full)
                print(f"💾 Cached FULL {split} dataset to {cache_file_full}")
            return dataset

            return dataset

    # ====== Default Transforms ======
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ====== Load or Build ======
    train_dataset = load_or_build(
        "train", train_transform, train_cache_samples, train_cache_full
    )
    val_dataset = load_or_build("val", val_transform, val_cache_samples, val_cache_full)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
