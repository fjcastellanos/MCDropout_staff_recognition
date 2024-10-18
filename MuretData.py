import torch
import utilsParameters
from PIL import  Image as PILImage

class Symbol():
    def __init__(self, boxes : list[int], position_in_staff :str, agnostic_symbol_type:str) -> None:
        self.box = boxes
        self.position_in_staff = position_in_staff
        self.agnostic_symbol_type = agnostic_symbol_type

    def __str__(self) -> str:
        return f'Boxes: {self.box}\t, Position: {self.position_in_staff}\t, Symbol: {self.agnostic_symbol_type}'

    def resize(self, ratio_width: float, ratio_height: float):
        box = self.box
        self.box = [int(box[0]/ratio_width), int(box[1]/ratio_height), int(box[2]/ratio_width), int(box[3]/ratio_height)]

        return self


class Region():
    def __init__(self, box: list[int], label: int, notes: list[Symbol] = None) -> None:
        self.box = box
        self.label = label
        self.notes = notes

    def __str__(self, symbols = True) -> str:
        message = f'Bounding boxes: {self.box}\t, Class: {utilsParameters.NUM_TO_CATEGORIES[self.label]}'

        if symbols:
            message += '\t, Symbols: '
            if self.notes is None:
                message += '[]'
            else:
                message += '\n\t\t['
                for sym in self.notes:
                    message += f'\n\t\t\t{sym}'
                message += '\n\t\t]'

        return message

    def isEmpty(self):
        return len(self.box) == 0

    def resize(self, ratio_width: float, ratio_height: float):

        box = self.box
        self.box = [int(box[0]/ratio_width), int(box[1]/ratio_height), int(box[2]/ratio_width), int(box[3]/ratio_height)]

        if self.notes is not None:
            self.notes = [note.resize(ratio_width=ratio_width, ratio_height=ratio_height) for note in self.notes]

        return self

class ImageExample():
    def __init__(self, regions: list[Region], image: PILImage, imageName: str, imagePath: str) -> None:
        self.regions = regions
        self.image: PILImage = image
        self.imageName = imageName
        self.imagePath = imagePath

    def isEmpty(self):
        return len(self.regions) <= 0

    def dataAsTensor(self):
        if self.isEmpty(): # If image does not contain any object (background)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(1, dtype=torch.int64)
        else:
            boxes = torch.as_tensor([region.box for region in self.regions], dtype=torch.float32) # [xmin, ymin, xmax, ymax]
            labels = torch.as_tensor([region.label for region in self.regions], dtype=torch.int64)

        return boxes, labels

    def get_boxes(self):
        return [region.box for region in self.regions]

    def get_name(self):
        return self.imageName

    def resize_examples(self, resize_shape):

        if resize_shape is not None:
            width, height = self.image.size
            ratio_width = width / resize_shape[0]
            ratio_height = height / resize_shape[1]

            self.image = self.image.resize(resize_shape)

            self.regions = [region for region in self.regions if not region.isEmpty()]
            self.regions = [region.resize(ratio_width=ratio_width, ratio_height=ratio_height) for region in self.regions]
            # for region in self.regions: region.resize(ratio_width=ratio_width, ratio_height=ratio_height)

    def __str__(self) -> str:
        for r in self.regions:
            print(r)