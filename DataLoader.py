
import MuretData
import utilsIO
import utilsParameters
from PIL import  ImageDraw
from PIL import  Image as PILImage
import torchvision.transforms as T

def load_dataset(imagesPath: list[str], jsonsPath: list[str], reshape = None):
    """_summary_

    Args:
        imagesPath (list[str]): paths to images
        jsonsPath (list[str]): paths to JSON data files
        reshape (list[int], optional): images size reshape if needed. Defaults to None.

    Returns:
        tuple[list[PILImage], list[list[Region]]]: lists of images and regions for each image
    """

    return [utilsIO.read_json_datafile(iPath, jPath, reshape) for iPath, jPath in zip(imagesPath, jsonsPath)]

def get_file_name(filePath: str):
    return filePath[filePath.rfind('/')+1:filePath.find('.')]

def draw_back_white_bb_image(height, width, boxes):
    targetImage = PILImage.new("RGB", (width, height), "#000000")

    draw = ImageDraw.Draw(targetImage)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], fill="#FFFFFF")

    return targetImage

def draw_boxes_on_image(example: MuretData.ImageExample, saveDirectory: str):
    """Save image with passes boxes drawn

    Args:
        image (Image): image to save
        boxes (list[list[int]]): bounding boxes to save
        imageName (str): image name
        saveDirectory (str): directory to save
    """
    # Read file & normalize image
    #img, regionsList = read_json_datafile(path_json, path_img, reshape)
    img = example.image

    print(saveDirectory)

    utilsIO.makeDirIfNotExist(saveDirectory)

    drawingImage = ImageDraw.Draw(img)

    # Save image
    # for box in example.get_boxes():
    #     drawingImage.rectangle(box, outline="red", width=3)
    [drawingImage.rectangle(box, outline="red") for box in example.get_boxes()]

    img.save(saveDirectory + example.imageName + '.jpg')

def get_transform():
    t = [
        T.ToTensor(),
        T.Grayscale(),
        ]

    return T.Compose(t)


def resize_box(box, vResize, hResize):
    """
      Boxes as [x1, y1, x2, y2]
    """
    if(len(box)) < 4:
      print(box)
    # Get dimensions
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1 # 100
    h = y2 - y1 # 100

    # Calculate resized dimensions
    w2 = w * hResize # 80
    h2 = h * vResize # 80

    # Get size difference
    wDif = w2 - w  # 100 - 80 = 20
    hDif = h2 - h  # 100 - 80 = 20

    # Get how much each coordinate must move
    xMov = wDif / 2
    yMov = hDif / 2

    # Get new coordinates
    x1 = x1 - xMov
    x2 = x2 + xMov

    y1 = y1 - yMov
    y2 = y2 + yMov

    return [x1, y1, x2, y2]

class DatasetLoader:
    def __init__(self, dataset_name: str, reshape:list[float] = None) -> None:
        """Create class that loads the dataset

        Args:
            dataset_name (str): forder path to dataset containing train, test and val's txt files without final / (Ex.: "datasets/Captitan")
            reshape: reshape of the images
        """
        self.datasetFolder = f'{utilsParameters.DRIVE_DATASETS_FOLDER}/{dataset_name}'
        self.reshape = reshape

    def loadTrainPaths(self):
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/train.txt')

    def loadTestPaths(self):
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/test.txt')

    def loadValPaths(self):
        return utilsIO.read_paths_dataset_files(f'{self.datasetFolder}/val.txt')

    def loadTrainDataset(self):
        trainJSonPaths, trainImagesPath = self.loadTrainPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(trainJSonPaths, trainImagesPath)]

    def loadValDataset(self):
        valJSonPaths, valImagesPath = self.loadValPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(valJSonPaths, valImagesPath)]

    def loadTestDataset(self):
        testJSonPaths, testImagesPath = self.loadTestPaths()
        return [utilsIO.read_json_datafile(json, img, self.reshape) for json, img in zip(testJSonPaths, testImagesPath)]

    def loadAllDatasetPaths(self):
        return self.loadTrainPaths(), self.loadValPaths(), self.loadTestPaths()

    def loadAllDataset(self):
        return self.loadTrainDataset(), self.loadValDataset(), self.loadTestDataset()

    def drawBoxesInDataset(self):
        train, val, test = self.loadAllDataset()
        [draw_boxes_on_image(example=example, saveDirectory=self.datasetFolder+"/train/") for example in train]
        [draw_boxes_on_image(example=example, saveDirectory=self.datasetFolder+"/val/") for example in val]
        [draw_boxes_on_image(example=example, saveDirectory=self.datasetFolder+"/test/") for example in test]