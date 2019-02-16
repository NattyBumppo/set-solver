import cv2
import numpy as np
import glob
import os

test_image = 'test.png'
template_image_dir = 'images/png/'

img_rgb = cv2.imread('test.jpg')

img_b = img_rgb.copy()
# set green and red channels to 0
img_b[:, :, 1] = 0
img_b[:, :, 2] = 0
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

img_g = img_rgb.copy()
# set blue and red channels to 0
img_g[:, :, 0] = 0
img_g[:, :, 2] = 0
img_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)

img_r = img_rgb.copy()
# set blue and green channels to 0
img_r[:, :, 0] = 0
img_r[:, :, 1] = 0
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

cv2.imwrite('g.png', img_g)
cv2.imwrite('b.png', img_b)
cv2.imwrite('r.png', img_r)

template_images = glob.glob(template_image_dir + '*.jpg')

template_images_base_filenames = [os.path.splitext(os.path.basename(p))[0] for p in template_images]

templates = []

for i, template_image in enumerate(template_images):
    template = cv2.imread(template_image)
    # Add separate templates for each color channel

    channels = []
    for channel in ['b', 'g', 'r']:
        template_channel = template.copy()

        if channel == 'b':
            template_channel[:, :, 1] = 0
            template_channel[:, :, 2] = 0
        elif channel == 'g':
            template_channel[:, :, 0] = 0
            template_channel[:, :, 2] = 0
        elif channel == 'r':
            template_channel[:, :, 0] = 0
            template_channel[:, :, 1] = 0

        template_channel = cv2.cvtColor(template_channel, cv2.COLOR_BGR2GRAY)

        channels.append(template_channel)

    templates.append((channels, template_images_base_filenames[i]))

font = cv2.FONT_HERSHEY_SIMPLEX

seen_cards = set()

for template_channels, name in templates:
    # print('Template matching for %s' % name)

    res_b = cv2.matchTemplate(img_b, template_channels[0], cv2.TM_CCOEFF_NORMED)    
    res_g = cv2.matchTemplate(img_g, template_channels[1], cv2.TM_CCOEFF_NORMED)
    res_r = cv2.matchTemplate(img_r, template_channels[2], cv2.TM_CCOEFF_NORMED)

    threshold = 0.7
    loc = np.where(np.logical_and(res_b >= threshold, res_g >= threshold, res_r >= threshold))

    w, h = template_channels[0].shape[::-1]

    matching_points = list(zip(*loc[::-1]))

    if len(matching_points) > 0:
        seen_cards.add(name)
        print('Match %s' % name)
    else:
        print('NO MATCH %s' % name)

    # for pt in zip(*loc[::-1]):
    #     print(pt)
    #     cv2.putText(img_rgb, name, (int(pt[0] + w / 2), int(pt[1] + h / 2)), font, 4, (255,255,255), 2, cv2.LINE_AA)

print('%s seen cards:' % len(seen_cards))
for c in seen_cards:
    print(c)

cv2.imwrite('res.png', img_rgb)
