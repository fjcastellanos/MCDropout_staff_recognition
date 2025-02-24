import torch
from typing import List
import DataLoaderOwn
import MuretData
import utilsIO
import utilsParameters

class TrainMusicDataset(torch.utils.data.Dataset):
    def __init__(self, name: str, jsonPaths: List[str], imagesPaths: List[str], resize_shape=None, transforms=DataLoaderOwn.get_transform(), box_resize_vertical=False, box_resize_horizontal=False):
        """
        Arguments:
            name: Name of the dataset (b-59-80, Seils, Mus-Trad-...)
            jsonPaths: List with all paths to json files
            imagesPaths: List with all paths to source images
        """
        self.name = name
        self.resize_shape = resize_shape
        self.jsonPaths = jsonPaths
        self.imagesPaths = imagesPaths
        self.transforms = transforms
        self.box_resize_vertical = box_resize_vertical
        self.box_resize_horizontal = box_resize_horizontal

    def __getitem__(self, idx):
        path_img = self.imagesPaths[idx]
        path_json = self.jsonPaths[idx]

        example: MuretData.ImageExample = utilsIO.read_json_datafile(path_json, path_img, self.resize_shape)
        example.resize_examples(self.resize_shape)

        boxes, labels = example.dataAsTensor()
        img = example.image

        draw_boxes = example.get_boxes()
        if self.box_resize_vertical or self.box_resize_horizontal:
            vResize = utilsParameters.BBOX_REDIMENSION if self.box_resize_vertical else 1
            hResize = utilsParameters.BBOX_REDIMENSION if self.box_resize_horizontal else 1
            draw_boxes = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in draw_boxes]

        target = {
            "boxes": boxes,
            "labels": labels,
            "name": example.get_name()
        }

        width, height = img.size
        targetImage = DataLoaderOwn.draw_back_white_bb_image(height=height, width=width, boxes=draw_boxes)

        if self.transforms is not None:
            img = self.transforms(img)
            targetImage = self.transforms(targetImage)

        return img, target, targetImage

    def getImagesPaths(self):
        return self.imagesPaths

    def __len__(self):
        return len(self.imagesPaths)

class TestMusicDataset(torch.utils.data.Dataset):
    def __init__(self, name: str, jsonPaths: List[str], imagesPaths: List[str], resize_shape=None, transforms=DataLoaderOwn.get_transform()):
        """
        Arguments:
            name: Name of the dataset (b-59-80, Seils, Mus-Trad-...)
            jsonPaths: List with all paths to json files
            imagesPaths: List with all paths to source images
        """
        self.name = name
        self.resize_shape = resize_shape
        self.jsonPaths = jsonPaths
        self.imagesPaths = imagesPaths
        self.transforms = transforms

    def __getitem__(self, idx):
        path_img = self.imagesPaths[idx]
        path_json = self.jsonPaths[idx]

        example: MuretData.ImageExample = utilsIO.read_json_datafile(path_json, path_img, self.resize_shape)
        img_orig = example
        example.resize_examples(self.resize_shape)

        boxes, _ = example.dataAsTensor()
        img = example.image

        if self.transforms is not None:
            img = self.transforms(img)

        return img, boxes, img_orig, path_img

    def getImagesPaths(self):
        return self.imagesPaths

    def __len__(self):
        return len(self.imagesPaths)

