import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from evogym import is_connected, has_actuator, sample_robot

def structure_to_tokens(structure) -> torch.Tensor:
    return torch.tensor(structure.flatten(), dtype=torch.long)

class RobotStructureDataset(Dataset):
    def __init__(self, data_dir, vocab, max_samples=-1, structure_shape=(5, 5), generate_if_empty=True, num_generated=1000):
        super().__init__()
        self.data_dir = data_dir
        self.structure_shape = structure_shape
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab.keys())}

        # Find all robot structure files (.npz) in the data directory.
        self.structure_files = []
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".npz"):
                        self.structure_files.append(os.path.join(root, file))

        if max_samples > 0:
            self.structure_files = self.structure_files[:max_samples]

        # If no files are found, generate new robot structures.
        if generate_if_empty and not self.structure_files:
            os.makedirs(data_dir, exist_ok=True)
            self.generate_samples(num_generated)

    def generate_samples(self, n, min_vox=6):
        # Desired distribution of voxel types (NOTE: chosen empirically)
        TARGET = {1: 0.25, 2: 0.15, 3: 0.30, 4: 0.30}
        counts = {k: 0 for k in TARGET}

        while len(self.structure_files) < n:
            robot, conns = sample_robot(self.structure_shape, pd=np.array([0.35, 0.2, 0.15, 0.15, 0.15]))

            # Ensure the robot is valid
            if not (is_connected(robot) and has_actuator(robot) and np.count_nonzero(robot) >= min_vox):
                continue

            # Check if robot would unbalance the distribution
            temp_counts = counts.copy()
            for k in temp_counts:
                temp_counts[k] += np.count_nonzero(robot == k)
            total_voxels = sum(temp_counts.values()) + 1e-9
            if any(temp_counts[k] / total_voxels > TARGET[k] * 1.3 for k in TARGET):
                continue

            # Save the new robot
            idx = len(self.structure_files)
            file_path = os.path.join(self.data_dir, f"robot_{idx:06d}.npz")
            np.savez(file_path, robot, conns)
            self.structure_files.append(file_path)

            for k in counts:
                counts[k] += np.count_nonzero(robot == k)

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx):
        with np.load(self.structure_files[idx]) as data:
            structure = data['arr_0']
            connections = data['arr_1']

        tokens = structure_to_tokens(structure)
        return {
            "tokens": tokens,
            "structure": torch.tensor(structure, dtype=torch.long),
            "connections": torch.tensor(connections),
            "seq_idx": torch.arange(len(tokens), dtype=torch.long),
        }

class TaskRobotDataset(Dataset):
    def __init__(self, data_dir, vocab, max_samples=-1, structure_shape=(5, 5)):
        super().__init__()
        self.data_dir = data_dir
        self.structure_shape = structure_shape
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab.keys())}

        # Load samples by reading output files and finding corresponding structures.
        self.samples = []
        for gen_folder in sorted(os.listdir(data_dir)):
            gen_path = os.path.join(data_dir, gen_folder)
            struct_dir = os.path.join(gen_path, "structure")
            output_file = os.path.join(gen_path, "output.txt")

            if not (os.path.isdir(struct_dir) and os.path.isfile(output_file)):
                continue

            with open(output_file, "r") as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        robot_id = int(parts[0])
                        score = float(parts[-1])
                        struct_file = os.path.join(struct_dir, f"{robot_id}.npz")

                        if os.path.exists(struct_file):
                            self.samples.append({"structure_file": struct_file, "score": score, "robot_id": robot_id})
                    except (ValueError, IndexError):
                        continue # Skip malformed lines.

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with np.load(sample["structure_file"]) as data:
            structure = data["structure"]
            connections = data["connections"]

        tokens = structure_to_tokens(structure)
        return {
            "tokens": tokens,
            "score": torch.tensor(sample["score"], dtype=torch.float).view(1),
            "structure": torch.tensor(structure, dtype=torch.long),
            "connections": torch.tensor(connections),
            "seq_idx": torch.arange(len(tokens), dtype=torch.long),
            "robot_id": sample["robot_id"],
        }

def custom_collate_fn(batch):
    if not batch:
        return {}

    batch_dict = {}
    first_elem = batch[0]

    for key in first_elem.keys():
        if key == "connections":
            continue

        values = [sample[key] for sample in batch]

        if key == "robot_id":
            batch_dict[key] = values
        elif isinstance(first_elem[key], torch.Tensor):
            batch_dict[key] = torch.stack(values)
        else:
            batch_dict[key] = values

    return batch_dict

def create_data_loaders(dataset, batch_size, train_split=0.9, shuffle=True, num_workers=4):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader