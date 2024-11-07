from typing import List, Tuple, Optional
import MuretData
import utilsIO
import utilsParameters
from PIL import ImageDraw
from PIL import Image as PILImage
import torchvision.transforms as T

def load_dataset(imagesPath: List[str], jsonsPath: List[str], reshape: Optional[List[int]] = None) -> List[MuretData.ImageExample]:
    """Load dataset by reading JSON data files and associated images.

    Args:
        imagesPath (List[str]): paths to images
        jsonsPath (List[str]): paths to JSON data files
        reshape (Optional[List[int]], optional): Reshape dimensions for images if needed. Defaults to None.

    Returns:
        List[MuretData.ImageExample]: List of ImageExample objects with images and regions.
    """
    return [utilsIO.read_json_datafile(iPath, jPath, reshape) for iPath, jPath in zip(imagesPath, jsonsPath)]

def get_file_name(filePath: str) -> str:
    return filePath[filePath.rfind('/')+1:filePath.find('.')]

def draw_back_white_bb_image(height: int, width: int, boxes: List[List[int]]) -> PILImage:
    """Create an image with bounding boxes drawn as white on a black background.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.
        boxes (List[List[int]]): List of bounding boxes.

    Returns:
        PILImage: Image with drawn bounding boxes.
    """
    targetImage = PILImage.new("RGB", (width, height), "#000000")
    draw = ImageDraw.Draw(targetImage)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], fill="#FFFFFF")
    return targetImage

def draw_boxes_on_image(example: MuretData.ImageExample, saveDirectory: str) -> None:
    """Save image with bounding boxes drawn on it.

    Args:
        example (MuretData.ImageExample): ImageExample containing image and boxes.
        saveDirectory (str): Directory to save the image.
    """
    img = example.image
    utilsIO.makeDirIfNotExist(saveDirectory)
    drawingImage = ImageDraw.Draw(img)
    [drawingImage.rectangle(box, outline="red") for box in example.get_boxes()]
    img.save(f"{saveDirectory}/{example.imageName}.jpg")

def get_transform():
    t = [
        T.ToTensor(),
        T.Grayscale(),
    ]
    return T.Compose(t)

def resize_box(box: List[int], vResize: float, hResize: float) -> List[float]:
    """Resize bounding box with given vertical and horizontal resizing factors.

    Args:
        box (List[int]): Bounding box coordinates as [x1, y1, x2, y2].
        vResize (float): Vertical resizing factor.
        hResize (float): Horizontal resizing factor.

    Returns:
        List[float]: Resized bounding box.
    """
    if len(box) < 4:
        print(box)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w, h = x2 - x1, y2 - y1
    w2, h2 = w * hResize, h * vResize
    wDif, hDif = w2 - w, h2 - h
    xMov, yMov = wDif / 2, hDif / 2
    return [x1 - xMov, y1 - yMov, x2 + xMov, y2 + yMov]

class DatasetLoader:
    def __init__(self, dataset_name: str, reshape: Optional[List[float]] = None) -> None:
        """Initialize dataset loader.

        Args:
            dataset_name (str): Folder path to dataset.
            reshape (Optional[List[float]]): Image reshape dimensions.
        """
        self.datasetFolder = f'{utilsParameters.DRIVE_DATASETS_FOLDER}/{dataset_name}'
        self.reshape = reshape

    def loadTrainPaths(self) -> Tuple[List[str], List[str]]:
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/train.txt')

    def loadTestPaths(self) -> Tuple[List[str], List[str]]:
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/test.txt')

    def loadValPaths(self) -> Tuple[List[str], List[str]]:
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/val.txt')

    def loadTrainDataset(self) -> List[MuretData.ImageExample]:
        trainJSonPaths, trainImagesPath = self.loadTrainPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(trainJSonPaths, trainImagesPath)]

    def loadValDataset(self) -> List[MuretData.ImageExample]:
        valJSonPaths, valImagesPath = self.loadValPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(valJSonPaths, valImagesPath)]

    def loadTestDataset(self) -> List[MuretData.ImageExample]:
        testJSonPaths, testImagesPath = self.loadTestPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(testJSonPaths, testImagesPath)]

    def loadAllDatasetPaths(self) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
        return self.loadTrainPaths(), self.loadValPaths(), self.loadTestPaths()

    def loadAllDataset(self) -> Tuple[List[MuretData.ImageExample], List[MuretData.ImageExample], List[MuretData.ImageExample]]:
        return self.loadTrainDataset(), self.loadValDataset(), self.loadTestDataset()

    def drawBoxesInDataset(self) -> None:
        train, val, test = self.loadAllDataset()
        [draw_boxes_on_image(example=example, saveDirectory=f"{self.datasetFolder}/train/") for example in train]
        [draw_boxes_on_image(example=example, saveDirectory=f"{self.datasetFolder}/val/") for example in val]
        [draw_boxes_on_image(example=example, saveDirectory=f"{self.datasetFolder}/test/") for example in test]

