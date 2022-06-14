import parse_args
import utils
from square_blue_signs import find_square_blue_signs
from red_circles import find_red_circles_signs
from stop_sign import find_stop_signs

args = parse_args.get_args()

image = utils.import_image(args.path)

# Detect red signs and labels them.
red_circles = find_red_circles_signs(image)
final_image = utils.label_sign(image, red_circles)

# Detect blue signs and labels.
blue_squares = find_square_blue_signs(image)
final_image = utils.label_sign(final_image, blue_squares)

# Detect stop signs.
stop_signs = find_stop_signs(image)
final_image = utils.label_sign(final_image, stop_signs)

utils.show_image(final_image, 'Final Image')
