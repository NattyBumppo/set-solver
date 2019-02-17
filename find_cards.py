import cv2
import numpy as np
import os
import imutils
import itertools
from line_iterator import createLineIterator

def not_near_any_saved_centers(cX, cY, saved_centers):
    min_dist_away_from_centers = 10

    for saved_center in saved_centers:
        if get_dist(saved_center, (cX, cY)) < min_dist_away_from_centers:
            return False

    return True

def get_contours_within_card(img, idx, approximate_card_area):
    card_img = img.copy()
    contour_img = img.copy()

    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    card_img = cv2.adaptiveThreshold(card_img, 180, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 5)

    # Dilate and erode to close holes (although we're working in inverse colors, so the code is actually an erosion and dilation)
    kernel = np.ones((6,6),np.uint8)
    card_img = cv2.morphologyEx(card_img, cv2.MORPH_OPEN, kernel)

    cnts = cv2.findContours(card_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts_and_moments = [(cnt, cv2.moments(cnt)) for cnt in cnts]
    # Sort by area (moment 00)
    cnts_and_moments.sort(key=lambda p: p[1]['m00'], reverse=True)

    # Bounds for the percentage of the card a shape should take up
    shape_card_percentage_lower = 0.05
    shape_card_percentage_higher = 0.22

    saved_contour_centers = []

    contours_to_keep = []
    for cnt, moment in cnts_and_moments:
        # Don't consider zero-area contours
        if moment['m00'] == 0:
            continue

        percentage_area = moment["m00"] / approximate_card_area

        cX = int((moment["m10"] / moment["m00"]))
        cY = int((moment["m01"] / moment["m00"]))

        if shape_card_percentage_lower < percentage_area < shape_card_percentage_higher:
            if not_near_any_saved_centers(cX, cY, saved_contour_centers):
                saved_contour_centers.append((cX, cY))
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2) # green contours mean accepted
                contours_to_keep.append(cnt)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (255, 0, 0), 2) # blue contours mean rejected due to center check
        else:
            cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 2) # red contours mean rejected due to size check

        # cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)

    # cv2.imwrite('out/thresh_inner_contours' + str(idx) + '.png', card_img)
    # cv2.imwrite('out/inner_contours' + str(idx) + '.png', contour_img)

    # Don't allow more than three shapes
    if len(contours_to_keep) <= 3:
        return contours_to_keep
    else:
        return []

def get_card_contours(img, flood_fill_size):
    # Resize image for faster processing
    resized = imutils.resize(img, width=800)
    resized_width = resized.shape[1]
    resized_height = resized.shape[0]
    resize_ratio = img.shape[0] / float(resized.shape[0])

    # Convert the resized image to grayscale, blur it slightly,
    # flood fill to ignore corner color background, and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Copy the flood-filled image
    flood = blurred.copy()

    cv2.floodFill(flood, None, (0, 0), 0, flood_fill_size, flood_fill_size)

    thresh = cv2.threshold(flood, 100, 255, cv2.THRESH_BINARY)[1]

    # cv2.imwrite('out/gray.png', gray)
    # cv2.imwrite('out/blurred.png', blurred)
    # cv2.imwrite('out/flood.png', flood)
    # cv2.imwrite('out/thresh.png', thresh)

    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    contours_with_area_percentages = [(c, cv2.moments(c)['m00'] / (resized_width * resized_height)) for c in cnts]
    valid_contours = [c for (c, p) in contours_with_area_percentages if p > 0.01]

    # Loop over the contours
    valid_orig_contours = []
    for i, c in enumerate(valid_contours):
        # Compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            # Skip zero-area contours
            continue
        cX = int((M["m10"] / M["m00"]) * resize_ratio)
        cY = int((M["m01"] / M["m00"]) * resize_ratio)
        area = M["m00"]

        # print('Contour %s: area %s of possible %s (%.3f)' % (i, area, resized_width * resized_height, area / (resized_width * resized_height)))

        # Multiply the contour (x, y)-coordinates by the resize ratio to get contours for the card in the original image
        c_orig = c.astype("float")
        c_orig *= resize_ratio
        c_orig = c_orig.astype("int")

        valid_orig_contours.append(c_orig)
     
        cv2.drawContours(img, [c_orig], -1, (0, 255, 0), 2)

    # cv2.imwrite('out/contours.png', img)
    
    return valid_orig_contours

def get_dist(pt0, pt1):
    return np.sqrt((pt0[0] - pt1[0]) * (pt0[0] - pt1[0]) + (pt0[1] - pt1[1]) * (pt0[1] - pt1[1]))

def get_debug_image(img, contours, card_colors, card_shapes, card_shadings, card_counts):

    debug_img = img.copy()

    for i, cnt in enumerate(contours):
        cv2.drawContours(debug_img, [cnt], -1, (0, 127, 0), 2)

        bounding_box = cv2.minAreaRect(cnt)
        bb_center, bb_size, angle = bounding_box

        font = cv2.FONT_HERSHEY_SIMPLEX
        bb_center_int = tuple([int(bb_center[0]), int(bb_center[1])])

        card_text = '(' + str(i) + ') ' + card_colors[i] + ' ' + card_shapes[i] + ' ' + card_shadings[i] + ' ' + str(card_counts[i])
        cv2.putText(debug_img, card_text, bb_center_int, font, 0.4, (144, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite('out/debug.png', debug_img)

    return debug_img

def get_card_color(masked_image, contour, long_line):
    b_acc = 0.0
    g_acc = 0.0
    r_acc = 0.0
    point_count = 0

    line_points, colors = createLineIterator(long_line[0], long_line[1], masked_image)

    # Use midpoint line for color average
    contiguous_non_white_pixel_groups = []
    contiguous_non_white_pixels_seen_recently = 0

    # Go through all of the colors on the horizontal midline
    for color_idx, color in enumerate(colors):
        coords = tuple(line_points[color_idx])

        # Only consider the point if it's actually inside the contour itself
        # (this will prevent literal "edge cases")
        pixels_within_contour_required = 3

        if cv2.pointPolygonTest(contour, coords, True) > pixels_within_contour_required:
            if color.any():
                # Check for white
                intensity_range = max(color) - min(color)
                # print(intensity_range)

                # Check for some deviation from the average color
                if intensity_range > 20:
                    b_acc += color[0]
                    g_acc += color[1]
                    r_acc += color[2]
                    point_count += 1
                    contiguous_non_white_pixels_seen_recently += 1
                    # print('color')
                # Little to no deviation from the average color indicates whiteness
                # else:
                #     print('WHITE')

                    # Log any non-white pixel strings we saw
                    if contiguous_non_white_pixels_seen_recently > 0:
                        contiguous_non_white_pixel_groups.append(contiguous_non_white_pixels_seen_recently)
                        contiguous_non_white_pixels_seen_recently = 0

        # else:
            # print('Point (%s, %s) not in contour' % (coords[0], coords[1]))

    contiguous_non_white_pixel_groups_as_percentage_of_horizontal = [g / len(line_points) for g in contiguous_non_white_pixel_groups]

    # print(contiguous_non_white_pixel_groups_as_percentage_of_horizontal)

    # Avoid divide-by-zero
    # print('Points within: %s' % point_count)
    if point_count == 0:
        point_count = 1
    b_mean = b_acc / point_count
    g_mean = g_acc / point_count
    r_mean = r_acc / point_count

    set_color = rgb_color_to_set_color(r_mean, g_mean, b_mean)

    # print('Average color (RGB): (%s, %s, %s)' % (r_mean, g_mean, b_mean))
    # print(set_color)

    return set_color, (b_mean, g_mean, r_mean)

def get_sample_contours():

    contours_to_return = []

    for filename in ['squiggle.jpg', 'oval.jpg', 'diamond.jpg']:
        img = cv2.imread(filename, 0)
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        thresh_debug_filename = 'out/' + os.path.splitext(filename)[0] + '_thresh_debug.jpg'
        cv2.imwrite(thresh_debug_filename, thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnt = cnts[0]

        contours_to_return.append(cnt)
        
        # Debug
        contour_img = img.copy()
        debug_filename = 'out/' + os.path.splitext(filename)[0] + '_debug.jpg'
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
        cv2.imwrite(debug_filename, contour_img)

    return contours_to_return


def get_card_types(test_image, contours):
    # Resize image for faster processing
    resized = imutils.resize(test_image, width=800)
    resized_width = resized.shape[1]
    resized_height = resized.shape[0]
    resize_ratio = test_image.shape[0] / float(resized.shape[0])

    resized_box = resized.copy()

    card_colors = ['U'] * len(contours)
    card_shapes = ['U'] * len(contours)
    card_shadings = ['U'] * len(contours)
    card_counts = [0] * len(contours)

    # squiggle_contour, oval_contour, diamond_contour = get_sample_contours()

    for i, c in enumerate(contours):
        # print('index ' + str(i))

        # Get resized contours
        c_resized = c.astype("float")
        c_resized /= resize_ratio
        c_resized = c_resized.astype("int")

        # Get resized bounding box of card
        resized_bounding_box = cv2.minAreaRect(c_resized)
        bb_center, bb_size, angle = resized_bounding_box

        if bb_size[0] < 10 or bb_size[1] < 10:
            # Reject very small cards
            continue

        # print('bb_center:' + str(bb_center))
        # print('bb_size:' + str(bb_size))
        # print('angle:' + str(angle))
        pts_raw = cv2.boxPoints(resized_bounding_box)
        pts_np = np.array([pts_raw], dtype=np.int32)
        
        pts_list = [tuple(pt) for pt in pts_raw]

        # Draw lines connecting points, for debugging
        cv2.line(resized_box, pts_list[0], pts_list[1], (0, 0, 255)) # Red
        cv2.line(resized_box, pts_list[1], pts_list[2], (255, 0, 0)) # Blue
        cv2.line(resized_box, pts_list[2], pts_list[3], (0, 255, 0)) # Green
        cv2.line(resized_box, pts_list[3], pts_list[0], (0, 255, 255)) # Yellow

        # Find center points
        cp0 = (int((pts_list[0][0] + pts_list[1][0]) / 2.0), int((pts_list[0][1] + pts_list[1][1]) / 2.0))
        cp1 = (int((pts_list[1][0] + pts_list[2][0]) / 2.0), int((pts_list[1][1] + pts_list[2][1]) / 2.0))
        cp2 = (int((pts_list[2][0] + pts_list[3][0]) / 2.0), int((pts_list[2][1] + pts_list[3][1]) / 2.0))
        cp3 = (int((pts_list[3][0] + pts_list[0][0]) / 2.0), int((pts_list[3][1] + pts_list[0][1]) / 2.0))

        # print('cp0 <--> cp2: %s' % get_dist(cp0, cp2))
        # print('cp1 <--> cp3: %s' % get_dist(cp1, cp3))

        # Get longer of the two center lines
        if get_dist(cp0, cp2) > get_dist(cp1, cp3):
            long_line = (cp0, cp2)
            short_line = (cp1, cp3)
        else:
            long_line = (cp1, cp3)
            short_line = (cp0, cp2)

        # Draw center lines
        cv2.line(resized_box, long_line[0], long_line[1], (0, 106, 255)) # Orange
        cv2.line(resized_box, short_line[0], short_line[1], (255, 0, 255)) # Magenta

        # Make a mask from the bounding box
        mask = np.zeros_like(resized)

        mask = cv2.fillPoly(mask, pts_np, (1, 1, 1))

        # cv2.imwrite('out/mask_' + str(i) + '.png', mask)

        # Mask out card
        resized_masked = resized * mask

        # cv2.imwrite('out/resized_masked_' + str(i) + '.png', resized_masked)

        # print(i)

        card_colors[i], mean_color_cv = get_card_color(resized_masked, c_resized, long_line)

        inner_card_contours = get_contours_within_card(resized_masked, i, bb_size[0] * bb_size[1])

        card_counts[i] = len(inner_card_contours)

        card_shadings[i] = get_card_shading(inner_card_contours, resized_masked, i, mean_color_cv, short_line, bb_center)

        card_shapes[i] = get_shape_from_contours(inner_card_contours, i)

    # cv2.imwrite('out/boxes.png', resized_box)

    return card_colors, card_shapes, card_shadings, card_counts

def get_shape_from_contours(inner_card_contours, idx):
    if len(inner_card_contours) == 0:
        # print('Error: no inner card contours for idx %s' % idx)
        return 'U'

    # Use first contour (they should all be similar anyway)
    cnt = inner_card_contours[0]

    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    if solidity < 0.93:
        # Squigglies have lower solidity
        # due to not matching their convex hulls well
        return 'S'

    # M = cv2.moments(inner_card_contours[0])
    # cX = int((M["m10"] / M["m00"]))
    # cY = int((M["m01"] / M["m00"]))
    # center = (cX, cY)

    bounding_box = cv2.minAreaRect(cnt)
    bb_center, bb_size, angle = bounding_box

    min_bounding_box_area = bb_size[0] * bb_size[1]

    # x,y,w,h = cv2.boundingRect(cnt)
    # rect_area = w*h
    # extent = float(area)/rect_area

    area_ratio = area / min_bounding_box_area

    # Ovals cover more of their minimum bounding rectangle
    if area_ratio > 0.7:
        # Oval
        return 'O'
    else:
        # Diamond
        return 'D'
    
    # (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)

    area_for_diamond = (MA * ma) / 2.0
    diff_from_diamond_area = abs(area - area_for_diamond)

    return '%.2f' % (diff_from_diamond_area / area_for_diamond)

def get_card_shading(inner_card_contours, resized_masked, idx, mean_color_cv, vertical_line_through_center_of_card, center_of_card):
    if len(inner_card_contours) == 0:
        # print('Error: no inner card contours for idx %s' % idx)
        return 'U'

    # print('Average color: ' + str(mean_color_cv))

    # Go through one of the contours vertically, looking for color changes
    M = cv2.moments(inner_card_contours[0])
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))

    card_center_to_first_point = (vertical_line_through_center_of_card[0][0] - center_of_card[0], vertical_line_through_center_of_card[0][1] - center_of_card[1])
    card_center_to_second_point = (vertical_line_through_center_of_card[1][0] - center_of_card[0], vertical_line_through_center_of_card[1][1] - center_of_card[1])

    first_line_point_through_contour_center = (int(cX + card_center_to_first_point[0]), int(cY + card_center_to_first_point[1]))
    second_line_point_through_contour_center = (int(cX + card_center_to_second_point[0]), int(cY + card_center_to_second_point[1]))

    line_img = resized_masked.copy()
 

    cv2.line(line_img, first_line_point_through_contour_center, second_line_point_through_contour_center, (0, 0, 255)) # Red

    # cv2.imwrite('out/line_through_contour' + str(idx) + '.png', line_img)

    gray = cv2.cvtColor(resized_masked, cv2.COLOR_BGR2GRAY)
    line_points, colors = createLineIterator(first_line_point_through_contour_center, second_line_point_through_contour_center, gray)

    line_length = get_dist(first_line_point_through_contour_center, second_line_point_through_contour_center)

    point_img = resized_masked.copy()
    
    contiguous_colored_pixel_groups = []
    contiguous_colored_pixels_seen_recently = 0

    # Go through all of the colors on the vertical midline
    used_colors = []
    white_intensity = 0
    for color_idx, color in enumerate(colors):
        coords = tuple(line_points[color_idx])

        # Only consider the point if it's actually inside the contour itself
        # (this will prevent literal "edge cases")
        pixels_within_contour_required = line_length * 0.02

        # Sample outside of the contour in order to find the white intensity for the card
        pixels_outside_of_contour_for_white_sample = line_length * 0.1

        if cv2.pointPolygonTest(inner_card_contours[0], coords, True) < -pixels_outside_of_contour_for_white_sample:
            white_intensity = max([color, white_intensity])
            # print('white intensity of %s' % white_intensity)

        if cv2.pointPolygonTest(inner_card_contours[0], coords, True) > pixels_within_contour_required:
            used_colors.append(color)
            if color > white_intensity - 25:
                
                cv2.circle(point_img, coords, 1, (255, 255, 255)) # white is white

                # Log any non-white pixel strings we saw
                if contiguous_colored_pixels_seen_recently > 0:
                    contiguous_colored_pixel_groups.append(contiguous_colored_pixels_seen_recently)
                    contiguous_colored_pixels_seen_recently = 0
            # Little to no deviation from the average color indicates whiteness
            else:
                contiguous_colored_pixels_seen_recently += 1
                cv2.circle(point_img, coords, 1, (0, 0, 255)) # color is red

    # Finally, collect any accumulated pixels
    if contiguous_colored_pixels_seen_recently > 0:
        contiguous_colored_pixel_groups.append(contiguous_colored_pixels_seen_recently)
        contiguous_colored_pixels_seen_recently = 0

    if len(used_colors) == 0:
        # Error occurred
        return 'U'

    range_val = max(used_colors) - min(used_colors)
    mean_val = np.mean(used_colors)
    # print('Range: %s' % range_val)
    # print('Mean: %s' % mean_val)

    # cv2.imwrite('out/line_through_contour_as_points' + str(idx) + '.png', point_img)

    # print(contiguous_colored_pixel_groups)

    if len(contiguous_colored_pixel_groups) > 0 and max(contiguous_colored_pixel_groups) > line_length * 0.5:
        if mean_val < 150:
            # filled shading
            shading = 'F'
        else:
            # striped shading
            shading = 'S'
    elif 0 <= len(contiguous_colored_pixel_groups) <= 2:
        # empty shading
        shading = 'E'
    elif len(contiguous_colored_pixel_groups) >= 4:
        # striped shading
        shading = 'S'
    else:
        # unknown shading
        shading = 'U'

    # print('Got shading of ' + shading)

    return shading

def adjust_contrast(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def rgb_color_to_set_color(r, g, b):
    # Scale color magnitudes first
    mag = np.sqrt(r * r + g * g + b * b)
    r_norm = r / mag
    g_norm = g / mag
    b_norm = b / mag

    # print(mag)
    # print('r: %s' % r)
    # print('g: %s' % g)
    # print('b: %s' % b)

    color_as_str = '(%.2f, %.2f, %.2f)' % (r, g, b)

    pixel = np.uint8([[[b,g,r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv = list(hsv.flatten())

    # print('hsv: %s' % hsv)
    # print('rgb: (%s, %s, %s), norms: (%s, %s, %s)' % (r, g, b, r_norm, g_norm, b_norm))

    if hsv[0] < 50:
        return 'R'
    elif hsv[0] < 90:
        return 'G'
    elif hsv[0] < 150:
        return 'P'
    else:    
        if g_norm > r_norm and g_norm > b_norm:
            return 'G' # green
        elif r > g * 1.1 and r > b * 1.1:
            return 'R' # red
        elif b_norm > g_norm:
            return 'P' # purple
        else:
            
            return 'U'

def is_set(a, b, c):
    return (((a[0] is not b[0] and b[0] is not c[0] and c[0] is not a[0]) or (a[0] is b[0] and b[0] is c[0] and c[0] is a[0])) and \
        ((a[1] is not b[1] and b[1] is not c[1] and c[1] is not a[1]) or (a[1] is b[1] and b[1] is c[1] and c[1] is a[1])) and \
        ((a[2] is not b[2] and b[2] is not c[2] and c[2] is not a[2]) or (a[2] is b[2] and b[2] is c[2] and c[2] is a[2])) and \
        ((a[3] is not b[3] and b[3] is not c[3] and c[3] is not a[3]) or (a[3] is b[3] and b[3] is c[3] and c[3] is a[3])))

def find_sets(card_colors, card_shapes, card_shadings, card_counts):
    
    indices = list(range(len(card_colors)))

    # Zip together cards on the table
    cards = []

    for card in zip(card_colors, card_shapes, card_shadings, card_counts, indices):
        cards.append(card)

    sets = []

    for combination in itertools.combinations(cards, 3):
        if is_set(combination[0], combination[1], combination[2]):
            sets.append(combination)

    return sets

def get_set_image(test_image, sets, contours, unknown_card_list):
    
    set_image = test_image.copy()

    colors = [ \
        # coral green
        (144, 255, 0), \
        # orange
        (0, 106, 255), \
        # dark green
        (0, 95, 0), \
        # yellow
        (0, 216, 255), \
        # purple
        (255, 0, 178), \
        # red
        (0, 0, 255), \
        # navy blue
        (95, 0, 0), \
        # magenta
        (215, 0, 215)
        ]

    set_accumulators_for_cards = [0] * len(contours)

    for i, set in enumerate(sets):
        for card_color, card_shape, card_shading, card_count, idx in set:
            set_accumulators_for_cards[idx] += 1

    # First, draw contours so known cards are shown as tracked
    for i, cnt in enumerate(contours):
        if i not in unknown_card_list:
            cv2.drawContours(set_image, [cnt], -1, (100, 100, 100), 1)

    # Draw sets
    for i, set in enumerate(sets):
        for card_color, card_shape, card_shading, card_count, idx in set:
            # Draw card contour in color, with an offset thickness to allow
            # multiple sets to be seen
            color_idx = i % len(colors)
            outline_color = colors[color_idx]
            min_thickness = 3

            thickness = min_thickness + (set_accumulators_for_cards[idx]-1)*10
            set_accumulators_for_cards[idx] -= 1

            # print('Drawing %s contour for %s with thickness %s' % (outline_color, idx, thickness))

            cnt = contours[idx]

            cv2.drawContours(set_image, [cnt], -1, outline_color, thickness)

    cv2.imwrite('out/set_image.png', set_image)

    return set_image

def get_unknown_card_list(card_colors, card_shapes, card_shadings, card_counts):
    unknown_card_list = []

    for i, (card_color, card_shape, card_shading, card_count) in enumerate(zip(card_colors, card_shapes, card_shadings, card_counts)):
        if card_color == 'U' or card_shape == 'card_shape' or card_shading == 'U' or card_count == 0:
            unknown_card_list.append(i)

    return unknown_card_list

def process_using_camera(mirror=False):
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)

        set_image, debug_image, card_count = process_image_for_set_data(img)

        if card_count == -1:
            # Show basic camera view (no sets to show)
            cv2.imshow('Set View', img)
            cv2.imshow('Debug View', img)
        else:
            # Show set view
            cv2.imshow('Set View', set_image)
            cv2.imshow('Debug View', debug_image)

        if cv2.waitKey(33) == 27: 
            break  # esc to quit
        if cv2.waitKey(33) == ord('s'): 
            # take screenshot
            cv2.imwrite('screenshot.png', img)
    cv2.destroyAllWindows()


def process_using_test_image():
    test_image_fn = 'test4.jpg'

    test_image = cv2.imread(test_image_fn)

    process_image_for_set_data(test_image)

def contour_sanity_check(contours):
    sane_contours = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area

        # print('%s solidity: %s' % (i, solidity))
        if solidity < 0.80:
            continue

        bounding_box = cv2.minAreaRect(cnt)
        bb_center, bb_size, angle = bounding_box

        # if bb_size[0] > bb_size[1]:
        #     long_side = bb_size[0]
        #     short_side = bb_size[1]
        # else:
        #     long_side = bb_size[1]
        #     short_side = bb_size[0]

        # side_ratio = bb_size[0] / bb_size[1]

        # print('%s side_ratio: %s' % (i, side_ratio))

        min_bounding_box_area = bb_size[0] * bb_size[1]

        area_ratio = area / min_bounding_box_area

        # print('%s area_ratio: %s' % (i, area_ratio))

        if area_ratio < 0.60:
            continue

        sane_contours.append(cnt)

    return sane_contours

def process_image_for_set_data(test_image):
    contour_sets = []

    flood_fill_size = 1

    while flood_fill_size < 5:
        contours = get_card_contours(test_image.copy(), flood_fill_size)
        # print('Got %s contours with ffs of %s' % (len(contours), flood_fill_size))
        contour_sets.append(contours)
        flood_fill_size += 1

    # Use max-length contour set
    contours = max(contour_sets, key=lambda c: len(c))

    if len(contours) <= 2 or len(contours) >= 16:
        print('Bad number of contours found')
        return None, None, -1

    contours = contour_sanity_check(contours)

    # Sort contours by area (so that they don't
    # jump around on the screen when visualized)
    contours.sort(key=lambda cnt: cv2.moments(cnt)['m00'], reverse=True)

    card_colors, card_shapes, card_shadings, card_counts = get_card_types(test_image, contours)

    unknown_card_list = get_unknown_card_list(card_colors, card_shapes, card_shadings, card_counts)

    debug_image = get_debug_image(test_image, contours, card_colors, card_shapes, card_shadings, card_counts)

    sets = find_sets(card_colors, card_shapes, card_shadings, card_counts)

    set_image = get_set_image(test_image, sets, contours, unknown_card_list)

    print('Found %s sets' % len(sets))
    print(sets)

    return set_image, debug_image, len(contours)


def main():
    test_mode = True

    if test_mode:
        process_using_test_image()
    else:
        process_using_camera()

if __name__ == '__main__':
    main()