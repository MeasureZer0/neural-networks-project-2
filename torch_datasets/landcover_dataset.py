# pyright: reportPrivateImportUsage=false
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torchvision.io as io
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_datasets.transforms import ValTransform


class LandcoverDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        split_file: Path,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ) -> None:

        self.image_dir = image_dir
        self.split_file = split_file
        self.return_meta = return_meta
        self.transform = transform

        with open(self.split_file, "r") as f:
            self.ids: List[str] = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        sample_id = self.ids[idx]

        image_path = self.image_dir / f"{sample_id}.jpg"
        mask_path = self.image_dir / f"{sample_id}_m.png"

        image_tensor = io.read_image(str(image_path))
        mask_tensor = io.read_image(str(mask_path))

        # CHW -> HWC, numpy uint8 - albumentations native format
        image_np: np.ndarray = image_tensor.permute(1, 2, 0).numpy()
        mask_np: np.ndarray = mask_tensor[0].numpy()

        if self.transform is not None:
            image, mask = self.transform(image_np, mask_np)
            mask = mask.long()
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask_np).long()

        mask = mask.squeeze(0)

        result = {
            "image": image,
            "mask": mask,
        }

        if self.return_meta:
            result["path"] = str(image_path)

        return result


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "landcover.ai.v1").resolve()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "landcover.ai.v1").resolve()

if __name__ == "__main__":
    dataset = LandcoverDataset(
        image_dir=DATA_DIR / "output",
        split_file=DATA_DIR / "val.txt",
        transform=ValTransform(),
    )
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"image: {batch['image'].shape} {batch['image'].dtype}")
    print(f"mask:  {batch['mask'].shape}  {batch['mask'].dtype}")
