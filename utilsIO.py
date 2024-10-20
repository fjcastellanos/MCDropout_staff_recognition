import MuretData
import utilsParameters
import DataLoaderOwn
import os
import json
from PIL import  Image as PILImage


def read_json_symbols(box: json, resize_shape, ratio_width, ratio_height) -> list[MuretData.Symbol]:
    """Reads the Symbols from the JSON and returns them all as a list

    Args:
        box (json): json following the symbols section
        resize_shape (list): None if no reshape is needed
        ratio_width (float): width ratio to resize
        ratio_height (float): height ratio to resize

    Returns:
        list[Symbol]: list of Symbols read
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
                xmin, xmax = int(round(xmin/ratio_width, 0)), int(round(xmax/ratio_width, 0))
                ymin, ymax = int(round(ymin/ratio_height, 0)), int(round(ymax/ratio_height, 0))

            listOfSymbols.append(MuretData.Symbol([xmin, ymin, xmax, ymax], symbolPos, symbolType))

    return listOfSymbols

def read_json_datafile(path_json: str, path_image: str, resize_shape = None, read_symbols = False) :
    """Reads the JSON data file and returns a list of Regions

    Args:
        path_json (str): path to JSON data file to read
        path_image (str): path to image to load
        resize_shape (list(int), optional): shape of reshape. Defaults to None.
        read_symbols (bool, optional): if Symbols must be read. Defaults to False.

    Returns:
        PILImage, list(Region): image and regions from that image
    """

    # print(f'Path_image: {path_image}')
    # print(f'Path_json: {filename}')

    image = PILImage.open(path_image)#.convert("RGB")
    width, height = image.size

    ratio_width, ratio_height = 0, 0
    if resize_shape is not None:
        image = image.resize(resize_shape)
        ratio_width = width / resize_shape[0]
        ratio_height = height / resize_shape[1]

    # getBoxesFunction = lambda b, rW, rH : [int(b['fromX']), int(b['fromY']), int(b['toX']), int(b['toY'])] if resize_shape is None else [int(b['fromX']/rW), int(b['fromY']/rH), int(b['toX']/rW), int(b['toY']/rH)]
    # getCategoriInt = lambda c : 'staff' if 'staff' in c else c
    # readSymbols = lambda b, r, rW, rH : read_json_symbols(box=b, resize_shape=r, ratio_width=rW, ratio_height=rH)
    # createRegionSymbols = lambda b, r, rH, rW : Region(box=getBoxesFunction(b=b['bounding_box'], rH=rH, rW=rW), label=getCategoriInt(b['type']), notes=readSymbols(b=b, r=r, rH=rH, rW=rW))
    # createRegionNonSymbols = lambda b, rH, rW : Region(box=getBoxesFunction(b=b['bounding_box'], rH=rH, rW=rW), label=getCategoriInt(b['type']), notes=None)

    boxes = []
    classes = []
    regionsList: list[MuretData.Region] = []
    with open(path_json) as f:
        # Read image from json (json info and image)
        example_dict = json.load(f)
        #filename_image = example_dict['filename'].encode('utf8')
        if "pages" not in example_dict:
            return MuretData.ImageExample(regions=[MuretData.Region(boxes, classes)], image=image, imageName=DataLoaderOwn.get_file_name(path_image), imagePath=path_image)

        pages = example_dict['pages']

        for _, page in enumerate(pages):
            if "regions" not in page:
                continue
            regions = page['regions']

            # if read_symbols:
            #     regionsList.append([createRegionSymbols(b=box, r=resize_shape, rH=ratio_height, rW=ratio_width) for box in regions])
            # else:
            #     regionsList.append([createRegionNonSymbols(b=box, rH=ratio_height, rW=ratio_width) for box in regions])

            for box in regions:
                box_data = box['bounding_box']
                category = box['type']
                if "staff" in category: # empty-staff => staff
                    category = "staff"

                # skip that aren't interesting categories (author, etc)
                if category in utilsParameters.CATEGORIES:
                    # print(f'region: {box}')
                    category_int = utilsParameters.CATEGORIES_TO_NUM[category]
                    xmin, xmax = box_data['fromX'], box_data['toX']
                    ymin, ymax = box_data['fromY'], box_data['toY']

                    if resize_shape is not None:
                        xmin, xmax = xmin/ratio_width, xmax/ratio_width
                        ymin, ymax = ymin/ratio_height, ymax/ratio_height

                    boxes_images = [int(xmin), int(ymin), int(xmax), int(ymax)]

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes_images = [int(box) for box in boxes_images]

                    if len(boxes_images) == 4:
                        if category == "staff" and read_symbols:
                            listOfSymbols = read_json_symbols(box=box, resize_shape=resize_shape, ratio_width=ratio_width, ratio_height=ratio_height)
                            regionsList.append(MuretData.Region(box = boxes_images, label = category_int, notes = listOfSymbols))
                        else:
                            regionsList.append(MuretData.Region(box = boxes_images, label = category_int))

    return MuretData.ImageExample(regions=regionsList, image=image, imageName=DataLoaderOwn.get_file_name(path_image), imagePath=path_image)

def read_paths_dataset_files(path_listJsons: str):
    """Read a file that contains a list to the paths to different JSON files

    Args:
        path_listJsons (str): path to file containing the list of json files

    Returns:
        [jsonList], [imagesList]: paths to different images and JSON data file
    """
    imagesList: list[str] = []
    jsonList: list[str] = []

    fileReading = open(path_listJsons, 'r')
    path = path_listJsons[:path_listJsons.rfind('/')+1]

    for file in fileReading:
        singleFileName = file[:file.rfind('.')]
        imagesList.append(path + 'images/' + singleFileName)
        jsonList.append(path + 'JSON/' + file[:file.rfind('n')+1])  # Removes empty spaces from end

    return jsonList, imagesList

def pathExist(dir_path):
    return os.path.exists(dir_path)

def makeDirIfNotExist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)