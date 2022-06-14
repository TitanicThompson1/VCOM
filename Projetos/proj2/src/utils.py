import glob
import xml.etree.ElementTree as et
import torch
import configparser
import os
import torchvision.transforms.functional as F

SRC_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_annotations(annotations_path):
    file_list = glob.glob(annotations_path + "*.xml")
    res = []
    for xml_file in file_list:
        info = parse_XML(xml_file)
        res.append(info)
    
    return res

def parse_XML(xml_file): 
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    info = {}
    objs = []

    for node in xroot:
        node_name = node.tag
        if node_name == 'folder':
            continue
        if node_name == 'filename':
            info['filename'] = node.text
        if node_name == 'size':
            # Get node children
            for child in node:
                if child.tag == 'width':
                    info['width'] = int(child.text)
                if child.tag == 'height':
                    info['height'] = int(child.text)
                if child.tag == 'depth':
                    info['depth'] = int(child.text)
        
        if node_name == 'object':
            obj = {}
            for child in node:
                if child.tag == 'name':
                    obj[child.tag] = child.text
                if child.tag == 'bndbox':
                    for box in child:
                        obj[box.tag] = int(box.text)
            objs.append(obj)

    info['objects'] = objs
    
    return info

def get_biggest_sign(objects):
    max_sign = ""
    max_area = 0
    for obj in objects:
        area = (obj['xmax'] - obj['xmin']) * (obj['ymax'] - obj['ymin'])
        if area > max_area:
            max_area = area
            max_sign = obj['name']
    return max_sign

def get_boxes(objects):
    res = []
    for obj in objects:
        res.extend([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
    
    return torch.tensor(res, dtype=torch.float)

def get_all_labels(data_loader):

    labels = torch.tensor([])
    for X, y in data_loader:
        labels = torch.cat((labels, y))
    return labels

def get_labels(objects):
    labels = []
    for obj in objects:
        labels.append(obj['name'])
    return labels

def get_areas(objs):
    """ Returns the areas of the bounding boxes, in a list
    """
    res = []
    for obj in objs:
        area = (obj['xmax'] - obj['xmin']) * (obj['ymax'] - obj['ymin'])
        res.append(area)
    return torch.tensor(res)

def get_configs(filepath):
    """
    Reads the project coonfigurations, such has max epochsa and learning rate, and saves it to a dict

    :param filepath: path to the file

    :return: dict with all configurations"""
    config = configparser.ConfigParser()
    config.read(os.path.join(SRC_DIR, filepath))
    res = {}
    for conf in config['DEFAULT']:
        value = config['DEFAULT'][conf]
        if value == 'yes' or value == 'no':
            res[conf] = config['DEFAULT'].getboolean(conf)
        elif value.isnumeric():
            res[conf] = config['DEFAULT'].getint(conf)
        elif value[0].isdigit():
            res[conf] = config['DEFAULT'].getfloat(conf)
        else:
            res[conf] = config['DEFAULT'][conf]

    assert res['multiclass']^ res['multilabel']^res['objectdetection'], "Multi-class, multi-label and object detection are mutually exclusive (there can only be one yes)."

    return res