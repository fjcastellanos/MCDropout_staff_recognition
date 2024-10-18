
import torch
import DataLoader
import MuretData
import utilsIO
import utilsParameters

class TrainMusicDataset(torch.utils.data.Dataset):
    def __init__(self, name, jsonPaths: list[str], imagesPaths: list[str], resize_shape = None, transforms = DataLoader.get_transform(), box_resize_vertical = False, box_resize_horizontal = False):
        """
        Arguments:
            name: Name of the dataset (b-59-80, Seils, Mus-Trad-...)
            path_json: list with all paths to json
            path_json: list with all paths to source images
        """
        self.name = name
        self.resize_shape = resize_shape
        self.jsonPaths: list[str] = jsonPaths
        self.imagesPaths: list[str] = imagesPaths
        self.transforms = transforms
        self.box_resize_vertical = box_resize_vertical
        self.box_resize_horizontal = box_resize_horizontal


    def __getitem__(self, idx):
        path_img = self.imagesPaths[idx]
        path_json = self.jsonPaths[idx]

        example: MuretData.ImageExample = utilsIO.read_json_datafile(path_json, path_img, self.resize_shape)

        # example: ImageExample = self.imagesPaths[idx]

        example.resize_examples(self.resize_shape)

        boxes, labels = example.dataAsTensor()
        img = example.image

        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        draw_boxes = example.get_boxes()
        if self.box_resize_vertical or self.box_resize_horizontal:
            vResize = utilsParameters.BBOX_REDIMENSION if self.box_resize_vertical else 1
            hResize = utilsParameters.BBOX_REDIMENSION if self.box_resize_horizontal else 1
            draw_boxes = [DataLoader.resize_box(box, vResize = vResize, hResize = hResize) for box in draw_boxes]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["name"]  = example.get_name()

        width, height = img.size
        targetImage = DataLoader.draw_back_white_bb_image(height=height, width=width, boxes=draw_boxes)

        if self.transforms is not None:
            img = self.transforms(img)
            targetImage = self.transforms(targetImage)

        return img, target, targetImage# torch.tensor([img]), [target]

    def getImagesPaths(self):
        return self.imagesPaths

    def __len__(self):
        return len(self.imagesPaths)

class TestMusicDataset(torch.utils.data.Dataset):
    def __init__(self, name, jsonPaths: list[str], imagesPaths: list[str], resize_shape = None, transforms = DataLoader.get_transform()):
        """
        Arguments:
            name: Name of the dataset (b-59-80, Seils, Mus-Trad-...)
            path_json: list with all paths to json
            path_json: list with all paths to source images
        """
        self.name = name
        self.resize_shape = resize_shape
        self.jsonPaths: list[str] = jsonPaths
        self.imagesPaths: list[str] = imagesPaths
        self.transforms = transforms


    def __getitem__(self, idx):
        path_img = self.imagesPaths[idx]
        path_json = self.jsonPaths[idx]

        example: MuretData.ImageExample = utilsIO.read_json_datafile(path_json, path_img, self.resize_shape)

        example.resize_examples(self.resize_shape)

        boxes, _ = example.dataAsTensor()
        img = example.image

        if self.transforms is not None:
            #img, target = self.transforms(img, target)
            img = self.transforms(img)

        return img, boxes# torch.tensor([img]), [target]

    def getImagesPaths(self):
        return self.imagesPaths

    def __len__(self):
        return len(self.imagesPaths)