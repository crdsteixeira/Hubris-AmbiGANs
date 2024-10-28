"""Module for calculating FOCD metric. Not used, to be deprecated."""

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.classifier.classifier_cache import ClassifierCache
from src.metrics.fid.fid import FID
from src.models import ConfigGAN
from src.utils.utility_functions import seed_worker


class FOCD(FID):
    """FOCD Metric."""

    def __init__(self, n_images: int, config: ConfigGAN, classifier_cache: ClassifierCache, dataset: Dataset) -> None:
        """Initialize class."""
        self.classifier_cache = classifier_cache
        dataloader = DataLoader(
            dataset,
            batch_size=config.train.step_2.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
        )

        dims = self.get_feature_map_fn(dataset[0][0].to(config.device), 0, 1).size(1)
        super().__init__(
            fid_stats_file=None,
            dims=dims,
            n_images=n_images,
            device=config.device,
            feature_map_fn=self.get_feature_map_fn,
        )

        # Calculate mu and sigma
        for batch in tqdm(dataloader):
            images = batch[0]
            if images.ndim < 2:
                raise ValueError(
                    f"Images must have at least two dimensions (batch size and channel), got {images.ndim}D tensor."
                )
            if images.shape[1] != 3:
                # Convert to RGB by repeating across the channel dimension
                images = images.repeat(1, 3, 1, 1)

            # Necessary to normalize pixels between 0 and 1
            self.fid.update((images + 1.0) / 2.0, is_real=True)

    def get_feature_map_fn(self, images: Tensor, batch_idx: int, batch_size: int) -> Tensor:
        """Get feature list of batch of images."""
        return self.classifier_cache.get(images, batch_idx, batch_size, output_feature_maps=True)[1]
