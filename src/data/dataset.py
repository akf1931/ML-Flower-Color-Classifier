from torch.utils.data import Dataset

class ColorLabeledDataset(Dataset):
    """Dataset wrapper pairing Flowers102 images with heuristic color labels."""

    def __init__(self, base_dataset, color_labels):
        """
        Args:
            base_dataset: Any dataset returning (image, original_label) pairs.
            color_labels (list[str]): One color label per image in base_dataset.
        """
        assert len(base_dataset) == len(color_labels), \
            "base_dataset and color_labels must have the same length."

        self.base = base_dataset
        self.color_labels = color_labels

        # Infer classes from labels
        self.classes = sorted(set(color_labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        label_name = self.color_labels[i]
        label = self.class_to_idx[label_name]
        return img, label

    def __repr__(self):
        return f"ColorLabeledDataset(n={len(self)}, classes={self.classes})"
