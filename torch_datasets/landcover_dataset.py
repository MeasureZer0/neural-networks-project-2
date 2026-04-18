from pathlib import Path
from typing import Callable, Dict, List, Optional

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io


class LandcoverDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        split_file: Path,
        transform: Optional[Callable] = None,
        return_meta: bool = False
    ) -> None:

        self.image_dir = image_dir
        self.split_file = split_file
        self.return_meta = return_meta
        self.transform = transform

        with open(self.split_file, "r") as f:
            self.ids: List[str] = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample_id = self.ids[idx]

        image_path = self.image_dir / f"{sample_id}.jpg"
        mask_path = self.image_dir / f"{sample_id}_m.png"

        image = io.read_image(str(image_path)).float() / 255.0
        mask = io.read_image(str(mask_path))

        if mask.shape[0] > 1:
            mask = mask[0]

        mask = mask.long()

        if self.transform is not None:
            image, mask = self.transform(image)

        result = {
            "image": image,
            "mask": mask,
        }

        if self.return_meta:
            result["path"] = str(image_path)

        return result
    
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / "landcover.ai.v1").resolve()
    
if __name__ == '__main__':

    dataset = LandcoverDataset(
        image_dir= DATA_DIR / "output",
        split_file= DATA_DIR / "val.txt",
        transform= None
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

