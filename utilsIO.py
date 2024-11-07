import MuretData
import utilsParameters
import DataLoaderOwn
import os
import json
from PIL import Image as PILImage
from typing import List

def read_json_symbols(box: json, resize_shape, ratio_width, ratio_height) -> List[MuretData.Symbol]:
    """Reads the Symbols from the JSON and returns them all as a list

    Args:
        box (json): json following the symbols section
        resize_shape (list): None if no reshape is needed
        ratio_width (float): width ratio to resize
        ratio_height (float): height ratio to resize

    Returns:
        List[Symbol]: list of Symbols read
    """
    listOfSymbols = None

    if 'symbols' in box:
        listOfSymbols = []
        symbols = box['symbols']

        for symbol in symbols:
            symbolBox = symbol['bounding_box']
            symbolType = symbol['agnostic_symbol_type']
            symbolPos = symbol['position_in_staff']

            xmin, xmax = symbolBox['fromX'], symbolBox['toX']
            ymin, ymax = symbolBox['fromY'], symbolBox['toY']

            if resize_shape is not None:
                xmin, xmax = int(round(xmin / ratio_width, 0)), int(round(xmax / ratio_width, 0))
                ymin, ymax = int(round(ymin / ratio_height, 0)), int(round(ymax / ratio_height, 0))

            listOfSymbols.append(MuretData.Symbol([xmin, ymin, xmax, ymax], symbolPos, symbolType))

    return listOfSymbols

def read_json_datafile(path_json: str, path_image: str, resize_shape=None, read_symbols=False):
    """Reads the JSON data file and returns a list of Regions

    Args:
        path_json (str): path to JSON data file to read
        path_image (str): path to image to load
        resize_shape (List[int], optional): shape of reshape. Defaults to None.
        read_symbols (bool, optional): if Symbols must be read. Defaults to False.

    Returns:
        PILImage, List[Region]: image and regions from that image
    """
    image = PILImage.open(path_image)
    width, height = image.size

    ratio_width, ratio_height = 0, 0
    if resize_shape is not None:
        image = image.resize(resize_shape)
        ratio_width = width / resize_shape[0]
        ratio_height = height / resize_shape[1]

    boxes = []
    classes = []
    regionsList: List[MuretData.Region] = []
    with open(path_json) as f:
        example_dict = json.load(f)
        if "pages" not in example_dict:
            return MuretData.ImageExample(regions=[MuretData.Region(boxes, classes)], image=image,
                                          imageName=DataLoaderOwn.get_file_name(path_image), imagePath=path_image)

        pages = example_dict['pages']

        for _, page in enumerate(pages):
            if "regions" not in page:
                continue
            regions = page['regions']

            for box in regions:
                box_data = box['bounding_box']
                category = box['type']
                if "staff" in category:
                    category = "staff"

                if category in utilsParameters.CATEGORIES:
                    category_int = utilsParameters.CATEGORIES_TO_NUM[category]
                    xmin, xmax = box_data['fromX'], box_data['toX']
                    ymin, ymax = box_data['fromY'], box_data['toY']

                    if resize_shape is not None:
                        xmin, xmax = xmin / ratio_width, xmax / ratio_width
                        ymin, ymax = ymin / ratio_height, ymax / ratio_height

                    boxes_images = [int(xmin), int(ymin), int(xmax), int(ymax)]

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes_images = [int(box) for box in boxes_images]

                    if len(boxes_images) == 4:
                        if category == "staff" and read_symbols:
                            listOfSymbols = read_json_symbols(box=box, resize_shape=resize_shape, ratio_width=ratio_width, ratio_height=ratio_height)
                            regionsList.append(MuretData.Region(box=boxes_images, label=category_int, notes=listOfSymbols))
                        else:
                            regionsList.append(MuretData.Region(box=boxes_images, label=category_int))

    return MuretData.ImageExample(regions=regionsList, image=image, imageName=DataLoaderOwn.get_file_name(path_image), imagePath=path_image)

def read_paths_dataset_files(path_listJsons: str):
    """Read a file that contains a list to the paths to different JSON files

    Args:
        path_listJsons (str): path to file containing the list of json files

    Returns:
        List[str], List[str]: paths to different images and JSON data file
    """
    imagesList: List[str] = []
    jsonList: List[str] = []

    fileReading = open(path_listJsons, 'r')
    path = path_listJsons[:path_listJsons.rfind('/') + 1]

    for file in fileReading:
        singleFileName = file[:file.rfind('.')]
        imagesList.append(path + 'images/' + singleFileName)
        jsonList.append(path + 'JSON/' + file[:file.rfind('n') + 1])

    return jsonList, imagesList

def pathExist(dir_path):
    return os.path.exists(dir_path)

def makeDirIfNotExist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

