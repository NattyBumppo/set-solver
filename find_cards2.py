import cv2
import numpy as np
import os
import imutils
from line_iterator import createLineIterator

def not_near_any_saved_centers(cX, cY, saved_centers):
    min_dist_away_from_centers = 10

    for saved_center in saved_centers:
        if get_dist(saved_center, (cX, cY)) < min_dist_away_from_centers:
            return False

    return True

def get_contours_within_card(img, idx, approximate_card_area, mean_color):

    card_img = img.copy()
    contour_img = img.copy()

    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # card_img = cv2.GaussianBlur(card_img, (1, 1), 0)

    card_img = cv2.adaptiveThreshold(card_img, 180, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 5)
    # card_img = cv2.threshold(card_img, 100, 180, cv2.THRESH_BINARY)[1]

    # Dilate and erode to close holes (although we're working in inverse colors, so the code is actually an erosion and dilation)
    kernel = np.ones((6,6),np.uint8)
    card_img = cv2.morphologyEx(card_img, cv2.MORPH_OPEN, kernel)

    cnts = cv2.findContours(card_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts_and_moments = [(cnt, cv2.moments(cnt)) for cnt in cnts]
    # Sort by area (moment 00)
    cnts_and_moments.sort(key=lambda p: p[1]['m00'], reverse=True)

    # Card area is the last 

    # card_area = cnts_and_moments[2][1]['m00']

    # Bounds for the percentage of the card a shape should take up
    shape_card_percentage_lower = 0.05
    shape_card_percentage_higher = 0.22

    saved_contour_centers = []

    contours_to_keep = []
    for cnt, moment in cnts_and_moments:
        # print('Moment: ' + str(moment['m00']))
        # Don't consider zero-area contours
        if moment['m00'] == 0:
            continue

        percentage_area = moment["m00"] / approximate_card_area

        cX = int((moment["m10"] / moment["m00"]))
        cY = int((moment["m01"] / moment["m00"]))

        if shape_card_percentage_lower < percentage_area < shape_card_percentage_higher:
            if not_near_any_saved_centers(cX, cY, saved_contour_centers):
                # print('Contour on ' + str(idx))
                # print(percentage_area)

                saved_contour_centers.append((cX, cY))
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2) # green contours mean accepted
                contours_to_keep.append(cnt)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (255, 0, 0), 2) # blue contours mean rejected due to center check
        else:
            # print('Rejected size ' + str(percentage_area))
            cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 2) # red contours mean rejected due to size check
        # print(cnt)
        # M = cv2.moments(cnt)
        
            # print(percentage_area)
        # Skip contours of weird size
        # if > percentage_area  0:
            
        #     continue

        # cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
    print('Contours? %s' % len(contours_to_keep))

    cv2.imwrite('out/thresh_inner_contours' + str(idx) + '.png', card_img)
    cv2.imwrite('out/inner_contours' + str(idx) + '.png', contour_img)

    # Don't allow more than three shapes
    if len(contours_to_keep) <= 3:
        return contours_to_keep
    else:
        return []

def get_card_contours(img):
    # Resize image for faster processing
    resized = imutils.resize(img, width=1000)
    resized_width = resized.shape[1]
    resized_height = resized.shape[0]
    resize_ratio = img.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # flood fill to ignore corner color background, and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # blurred = gray

    # corner_color = blurred[0, 0]

    # Copy the flood-filled image
    flood = blurred.copy()

    # Flood fill mask
    # h, w = flood.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(flood, None, (0, 0), 0, 1, 1)

    thresh = cv2.threshold(flood, 100, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite('out/gray.png', gray)
    cv2.imwrite('out/blurred.png', blurred)
    cv2.imwrite('out/flood.png', flood)
    cv2.imwrite('out/thresh.png', thresh)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    contours_with_area_percentages = [(c, cv2.moments(c)['m00'] / (resized_width * resized_height)) for c in cnts]
    valid_contours = [c for (c, p) in contours_with_area_percentages if p > 0.01]

    # loop over the contours
    valid_orig_contours = []
    for i, c in enumerate(valid_contours):
        # compute the center of the contour, then detect the name of the
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

    cv2.imwrite('out/contours.png', img)
    
    return valid_orig_contours

def get_dist(pt0, pt1):
    # print('Getting dist between ' + str(pt0) + ' and ' + str(pt1))

    return np.sqrt((pt0[0] - pt1[0]) * (pt0[0] - pt1[0]) + (pt0[1] - pt1[1]) * (pt0[1] - pt1[1]))

def debug_draw_card_types(img, contours, card_colors, card_shapes, card_fills, card_counts):

    debug_img = img.copy()

    for i, cnt in enumerate(contours):
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)

        bounding_box = cv2.minAreaRect(cnt)
        bb_center, bb_size, angle = bounding_box

        font = cv2.FONT_HERSHEY_SIMPLEX
        bb_center_int = tuple([int(bb_center[0]), int(bb_center[1])])
        # bb_corner = tuple([bb_center_int[0] - int(bb_size[0]/2.0), bb_center_int[1] + int(bb_size[1]/2.0)])
        card_text = str(i) + ' ' + card_colors[i] + ' ' + card_shapes[i] + ' ' + card_fills[i] + ' ' + str(card_counts[i])
        cv2.putText(debug_img, card_text, bb_center_int, font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('out/debug.png', debug_img)

def get_card_color(masked_image, contour, long_line):
    b_acc = 0.0
    g_acc = 0.0
    r_acc = 0.0
    point_count = 0

    line_points, colors = createLineIterator(long_line[0], long_line[1], masked_image)
    # print('long_line:')
    # print(long_line)
    # print(colors)

    # Use midpoint line for color average
    continguous_non_white_pixel_groups = []
    continguous_non_white_pixels_seen_recently = 0

    # Go through all of the colors on the horizontal midline
    for color_idx, color in enumerate(colors):
        coords = tuple(line_points[color_idx])

        # Only consider the point if it's actually inside the contour itself
        # (this will prevent literal "edge cases")
        pixels_within_contour_required = 3

        if cv2.pointPolygonTest(contour, coords, True) > pixels_within_contour_required:
            if color.any():
                # Check for white
                
                # print('--')
                # # print(color)
                # print(color[0])
                # print(color[1])
                # print(color[2])
                sum_intensity = float(color[0]) + float(color[1]) + float(color[2])
                # print(sum_intensity)

                avg_intensity = sum_intensity / 3.0
                # print(avg_intensity)

                # print(type(avg_intensity))
                # Check for some deviation from the average color
                if abs(color[0] - avg_intensity) > 10 or abs(color[1] - avg_intensity) > 10 or abs(color[2] - avg_intensity) > 10:
                    b_acc += color[0]
                    g_acc += color[1]
                    r_acc += color[2]
                    point_count += 1
                    continguous_non_white_pixels_seen_recently += 1
                    # print('color')
                # Little to no deviation from the average color indicates whiteness
                # else:
                #     print('WHITE')

                    # Log any non-white pixel strings we saw
                    if continguous_non_white_pixels_seen_recently > 0:
                        continguous_non_white_pixel_groups.append(continguous_non_white_pixels_seen_recently)
                        continguous_non_white_pixels_seen_recently = 0

        # else:
            # print('Point (%s, %s) not in contour' % (coords[0], coords[1]))
            # print()

    continguous_non_white_pixel_groups_as_percentage_of_horizontal = [g / len(line_points) for g in continguous_non_white_pixel_groups]

    # print(continguous_non_white_pixel_groups_as_percentage_of_horizontal)

    # b, g, r, _ = (cv2.mean(resized_masked))
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



    # Put colors onto cards
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bb_center_int = tuple([int(bb_center[0]), int(bb_center[1])])
    # bb_corner = tuple([bb_center_int[0] - int(bb_size[0]/2.0), bb_center_int[1] + int(bb_size[1]/2.0)])
    # cv2.putText(resized_box, str(i) + ' ' + set_color, bb_center_int, font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)


def get_card_types(test_image, contours):
    # Resize image for faster processing
    resized = imutils.resize(test_image, width=800)
    resized_width = resized.shape[1]
    resized_height = resized.shape[0]
    resize_ratio = test_image.shape[0] / float(resized.shape[0])

    resized_box = resized.copy()

    card_colors = ['U'] * len(contours)
    card_shapes = ['U'] * len(contours)
    card_fills = ['U'] * len(contours)
    card_counts = [0] * len(contours)

    for i, c in enumerate(contours):
        print('index ' + str(i))

        # Get resized contours
        c_resized = c.astype("float")
        c_resized /= resize_ratio
        c_resized = c_resized.astype("int")

        # Get resized bounding box of card
        resized_bounding_box = cv2.minAreaRect(c_resized)
        bb_center, bb_size, angle = resized_bounding_box

        # print(bb_size)
        if bb_size[0] < 10 or bb_size[1] < 10:
            # Reject very small cards
            continue

        # print('bb_center:' + str(bb_center))
        # print('bb_size:' + str(bb_size))
        # print('angle:' + str(angle))
        pts_raw = cv2.boxPoints(resized_bounding_box)
        pts_np = np.array([pts_raw], dtype=np.int32)
        # pts_np = np.array([[0,0],[100,100], [200,100], [100,200]], np.int32)
        
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

        # Get longer of the two center lines
        # print('cp0 <--> cp2: %s' % get_dist(cp0, cp2))
        # print('cp1 <--> cp3: %s' % get_dist(cp1, cp3))

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

        cv2.imwrite('out/resized_masked_' + str(i) + '.png', resized_masked)

        card_colors[i], mean_color_cv = get_card_color(resized_masked, c_resized, long_line)

        card_contours = get_contours_within_card(resized_masked, i, bb_size[0] * bb_size[1], mean_color_cv)

        card_counts[i] = len(card_contours)
        

    cv2.imwrite('out/boxes.png', resized_box)

    return card_colors, card_shapes, card_fills, card_counts

def rgb_color_to_set_color(r, g, b):
    # Scale color magnitudes first
    mag = np.sqrt(r * r + g * g + b * b)
    r /= mag
    g /= mag
    b /= mag

    # print(mag)
    # print('r: %s' % r)
    # print('g: %s' % g)
    # print('b: %s' % b)

    color_as_str = '(%.2f, %.2f, %.2f)' % (r, g, b)

    # return color_as_str

    if g > r and g > b:
        return 'G' # green
    elif r > g and r > b:
        return 'R' # red
    elif b > r and b > g:
        return 'P' # purple
    else:
        return 'U'

def main():
    test_image_fn = 'test3.jpg'

    test_image = cv2.imread(test_image_fn)

    contours = get_card_contours(test_image.copy())

    card_colors, card_shapes, card_fills, card_counts = get_card_types(test_image, contours)

    debug_draw_card_types(test_image, contours, card_colors, card_shapes, card_fills, card_counts)


if __name__ == '__main__':
    main()

