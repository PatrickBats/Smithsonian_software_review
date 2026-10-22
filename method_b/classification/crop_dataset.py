"""Training-only augmentation and normalized specimen crop datasets."""
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class Crops(Dataset):
    def __init__(self, root, rows, labels, augment):
        """Initialize the classifier model or crop dataset with the recorded configuration.

        Args:
            root: Root directory containing the prepared crop dataset.
            rows: Specimen manifest records, including record IDs, labels and split membership.
            labels: Class names in checkpoint output order.
            augment: Whether to apply training-only quarter-turns and reflections.

        Returns:
            None: Initializes this instance.
        """
        self.rows = rows
        self.targets = [labels.index(r["sponsor_category"]) for r in rows]
        self.augment = augment
        self.images = []
        for row in rows:
            with Image.open(root / row["image_relpath"]) as im:
                side = max(im.size)
                im = ImageOps.pad(im, (side, side), color="white")
                self.images.append(im.resize((224, 224), Image.Resampling.BICUBIC).copy())

    def __len__(self):
        """Return the number of classifier crop records.

        Returns:
            int: Number of specimens.
        """
        return len(self.rows)

    def __getitem__(self, index):
        """Prepare one crop using the recorded augmentation and ImageNet normalization.

        Args:
            index: Zero-based specimen index.

        Returns:
            tuple: Image tensor (3, 224, 224), class index, and specimen index.
        """
        im = self.images[index]
        if self.augment:
            # Quarter-turns and reflections retain the full square specimen crop.
            k = random.randrange(4)
            if k:
                im = im.transpose(
                    [
                        Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180,
                        Image.Transpose.ROTATE_270,
                    ][k - 1]
                )
            if random.random() < 0.5:
                im = ImageOps.mirror(im)
        x = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x, self.targets[index], index
