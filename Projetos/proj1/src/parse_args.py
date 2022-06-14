import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Identify the traffics signs pressent on a given image.')
    parser.add_argument('path', help='Path to the image', type=str)
    args = parser.parse_args()
    return args