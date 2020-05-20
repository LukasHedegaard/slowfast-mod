from pathlib import Path
import shutil
import re
from tqdm.auto import tqdm
from typing import Union
from src.utils.download import download_and_extract_archive
from argparse import ArgumentParser

# from torch.utils.data import random_split
# from torchvision.datasets import UCF101, VisionDataset
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
# def ucf101(
#     data_path=DATA_PATH,
#     annotation_path=SPLITS_PATH,
#     val_split=0.05,
#     num_frames=16,
#     clip_steps=50,
#     fold=1,
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

#     test = UCF101(
#         data_path,
#         annotation_path,
#         num_frames,
#         step_between_clips=clip_steps,
#         fold=fold,
#         train=False,
#         transform=test_tfms,
#         num_workers=NUM_CPU,
#     )

#     train_val = UCF101(
#         data_path,
#         annotation_path,
#         num_frames,
#         step_between_clips=clip_steps,
#         fold=fold,
#         train=True,
#         transform=train_tfms,
#         num_workers=NUM_CPU,
#     )

#     total_train_val_samples = len(train_val)

#     total_val_samples = round(val_split * total_train_val_samples)

#     train, val = random_split(
#         train_val, [total_train_val_samples - total_val_samples, total_val_samples]
#     )

#     return train, val, test


def download_ucf101(root_path: Union[Path, str]):
    ROOT_PATH = Path(root_path)
    DATA_PATH = ROOT_PATH / "data"
    SPLITS_PATH = ROOT_PATH / "splits"

    # splits
    if not SPLITS_PATH.exists():
        download_and_extract_archive(
            url="https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-DetectionTask.zip",
            download_root=ROOT_PATH,
            extract_root=ROOT_PATH,
            md5="b1502550fd32ce2c1649167d49c1c65d",
            remove_finished=False,
        )
        (ROOT_PATH / "UCF101_Action_detection_splits").rename(SPLITS_PATH)

    # data
    if not DATA_PATH.exists():
        download_and_extract_archive(
            url="http://storage.googleapis.com/thumos14_files/UCF101_videos.zip",
            download_root=ROOT_PATH,
            extract_root=ROOT_PATH,
            # md5="2b5850aae21b1627ef408e176c82a7a0",
        )
        (ROOT_PATH / "UCF101").rename(DATA_PATH)


def restructure_ucf101_data(
    source_path: Union[Path, str], target_path: Union[Path, str], move=False
):
    """Copies data from `source_path` to `target_path` in a structure mathcing the original UCF-101 dataset splits annotations

    Arguments:
        source_path {Path} -- path to data (from THUMOS source)
        target_path {Path} -- path to data (new location)
    """
    labels = set()
    source_path = Path(source_path)
    target_path = Path(target_path)

    print(
        "{} data with old structure from {} to {}".format(
            "Moving" if move else "Copying", str(source_path), str(target_path)
        )
    )
    source_vid_paths = list(source_path.glob("*.avi"))

    with tqdm(total=len(source_vid_paths)) as pbar:
        for vid in source_vid_paths:
            try:
                label = re.search("v_(.+?)_", str(vid.name)).group(1)
                label_folder_path = target_path / label
                new_vid_path = label_folder_path / vid.name

                if label not in labels:
                    labels.add(label)
                    label_folder_path.mkdir(parents=True, exist_ok=True)

                if move:
                    vid.rename(new_vid_path)
                else:  # copy
                    shutil.copy(vid, new_vid_path)

            except AttributeError:
                print(
                    "Unable to move ''{}'' because doesn't match the file pattern `v_XXX_*.avi`".format(
                        vid.name
                    )
                )
            pbar.update(1)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--path", default="~/datasets/ucf101", type=str)
    args = parser.parse_args()

    dataset_path = Path(args.path)
    download_ucf101(dataset_path)
    restructure_ucf101_data(
        dataset_path / "data", dataset_path / "data", move=True,
    )
