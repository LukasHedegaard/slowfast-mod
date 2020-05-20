from functools import partial
from multiprocessing import Pool
from typing import Union
from src.utils.download import (
    download_and_extract_archive,
    download_url,
    extract_archive,
)
from pathlib import Path
from argparse import ArgumentParser

# import warnings

# from torch.utils.data import random_split
# from torchvision.datasets import HMDB51, VisionDataset
# from torchvision.transforms import Compose

# from src.utils import DATASETS_PATH, NUM_CPU
# from src.utils.cache import memoize
# from src.utils.transforms import (
#     CenterCrop,
#     Normalize,
#     RandomCrop,
#     RandomHorizontalFlip,
#     Resize,
#     ToFloatTensorInZeroOne,
# )

# @memoize()
# def hmdb51(
#     data_path=DATA_PATH,
#     annotation_path=SPLITS_PATH,
#     val_split=0.05,
#     num_frames=16,
#     clip_steps=50,
#     fold=1,
#     transform=True,
# ) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:

#     # Modified from setup in https://github.com/pytorch/vision/blob/master/references/video_classification/train.py
#     train_tfms = Compose(
#         [
#             ToFloatTensorInZeroOne(),
#             Resize((128, 171)),
#             RandomHorizontalFlip(),
#             Normalize(
#                 mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
#             ),
#             RandomCrop((112, 112)),
#         ]
#     )
#     test_tfms = Compose(
#         [
#             ToFloatTensorInZeroOne(),
#             Resize((128, 171)),
#             Normalize(
#                 mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
#             ),
#             CenterCrop((112, 112)),
#         ]
#     )

#     with warnings.catch_warnings():
#         # Silence: python3.7/site-packages/torchvision/io/video.py:106: UserWarning: The pts_unit 'pts' gives wrong results and will be removed in a follow-up version. Please use pts_unit 'sec'.
#         warnings.simplefilter("ignore")

#         test = HMDB51(
#             data_path,
#             annotation_path,
#             num_frames,
#             step_between_clips=clip_steps,
#             fold=fold,
#             train=False,
#             transform=test_tfms if transform else None,
#             num_workers=NUM_CPU,
#         )

#         train_val = HMDB51(
#             data_path,
#             annotation_path,
#             num_frames,
#             step_between_clips=clip_steps,
#             fold=fold,
#             train=True,
#             transform=train_tfms if transform else None,
#             num_workers=NUM_CPU,
#         )

#     total_train_val_samples = len(train_val)

#     total_val_samples = round(val_split * total_train_val_samples)

#     train, val = random_split(
#         train_val, [total_train_val_samples - total_val_samples, total_val_samples]
#     )

#     return train, val, test


def download_hmdb51(root_path: Union[Path, str]):
    ROOT_PATH = Path(root_path)
    DATA_PATH = ROOT_PATH / "data"
    SPLITS_PATH = ROOT_PATH / "splits"

    # splits
    if not SPLITS_PATH.exists():
        download_and_extract_archive(
            url="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
            download_root=ROOT_PATH,
            extract_root=ROOT_PATH,
            md5="15e67781e70dcfbdce2d7dbb9b3344b5",
            remove_finished=True,
        )
        (ROOT_PATH / "testTrainMulti_7030_splits").rename(SPLITS_PATH)

    # data
    if not DATA_PATH.exists():
        download_url(
            url="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
            root=ROOT_PATH,
            md5="517d6f1f19f215c45cdd4d25356be1fb",
        )

        extract_archive(
            from_path=ROOT_PATH / "hmdb51_org.rar",
            to_path=DATA_PATH,
            remove_finished=True,
        )
        Pool().map(
            partial(extract_archive, remove_finished=True), DATA_PATH.glob("*.rar")
        )


# @memoize()
# def hmdb51(
#     data_path=DATA_PATH,
#     annotation_path=SPLITS_PATH,
#     val_split=0.05,
#     num_frames=16,
#     clip_steps=50,
#     fold=1,
#     transform=True,
# ) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:

#     # Modified from setup in https://github.com/pytorch/vision/blob/master/references/video_classification/train.py
#     train_tfms = Compose(
#         [
#             ToFloatTensorInZeroOne(),
#             Resize((128, 171)),
#             RandomHorizontalFlip(),
#             Normalize(
#                 mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
#             ),
#             RandomCrop((112, 112)),
#         ]
#     )
#     test_tfms = Compose(
#         [
#             ToFloatTensorInZeroOne(),
#             Resize((128, 171)),
#             Normalize(
#                 mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
#             ),
#             CenterCrop((112, 112)),
#         ]
#     )

#     with warnings.catch_warnings():
#         # Silence: python3.7/site-packages/torchvision/io/video.py:106: UserWarning: The pts_unit 'pts' gives wrong results and will be removed in a follow-up version. Please use pts_unit 'sec'.
#         warnings.simplefilter("ignore")

#         test = HMDB51(
#             data_path,
#             annotation_path,
#             num_frames,
#             step_between_clips=clip_steps,
#             fold=fold,
#             train=False,
#             transform=test_tfms if transform else None,
#             num_workers=NUM_CPU,
#         )

#         train_val = HMDB51(
#             data_path,
#             annotation_path,
#             num_frames,
#             step_between_clips=clip_steps,
#             fold=fold,
#             train=True,
#             transform=train_tfms if transform else None,
#             num_workers=NUM_CPU,
#         )

#     total_train_val_samples = len(train_val)

#     total_val_samples = round(val_split * total_train_val_samples)

#     train, val = random_split(
#         train_val, [total_train_val_samples - total_val_samples, total_val_samples]
#     )

#     return train, val, test


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--path", default="/Users/au478108/Projects/datasets/hmdb51", type=str
    )
    args = parser.parse_args()

    dataset_path = Path(args.path)
    download_hmdb51(dataset_path)
