# import cv2
# import threading
# import numpy as np
#
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
#
# def get_matching_pos(img, template):
#     # get the position of the most similar patch to template in img and compute the similarity
#     res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     cv2.imwrite('/Users/xiaohan/Desktop/images/t.tif', template)
#     cv2.imwrite('/Users/xiaohan/Desktop/images/s.tif', img[max_loc[0]:max_loc[0]+50, max_loc[1]:max_loc[1]+50, :])
#     return max_loc, max_val
#
#
# def compute_patch_similarity(s_img, t_img, patch_size=50):
#     """
#     Split target image into small patch. Find the most similar patch in source image to each target patch.
#     Compute the similarity for each target-source pair
#     :param s_img: source iamge
#     :param t_img: target image
#     :param patch_size: patch size
#     :return: similarity matrix
#     """
#     s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
#     t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
#     sobel_s_x = cv2.Sobel(s_img, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_s_y = cv2.Sobel(s_img, cv2.CV_64F, 0, 1, ksize=5)
#     sobel_s = np.abs(sobel_s_x) + np.abs(sobel_s_y)
#     sobel_s = sobel_s / sobel_s.max() * 255
#     sobel_s = sobel_s.astype(np.uint8)
#     _, sobel_s = cv2.threshold(sobel_s, 127, 255, cv2.THRESH_BINARY)
#     # _, sobel_s = cv2.threshold(sobel_s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     sobel_t_x = cv2.Sobel(t_img, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_t_y = cv2.Sobel(t_img, cv2.CV_64F, 0, 1, ksize=5)
#     sobel_t = np.abs(sobel_t_x) + np.abs(sobel_t_y)
#     sobel_t = sobel_t / sobel_t.max() * 255
#     sobel_t = sobel_t.astype(np.uint8)
#     _, sobel_t = cv2.threshold(sobel_t, 127, 255, cv2.THRESH_BINARY)
#     # _, sobel_t = cv2.threshold(sobel_t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imwrite('/Users/xiaohan/Desktop/images/shanghai2_grad.tif', sobel_s)
#     cv2.imwrite('/Users/xiaohan/Desktop/images/vegas_grad.tif', sobel_t)
#
#     height = s_img.shape[0]
#     h_num = height // patch_size
#     similarity = np.zeros([h_num, h_num, 3])
#
#     for h in range(h_num):
#         for w in range(h_num):
#             t_patch = sobel_t[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size]
#             top_left, value = get_matching_pos(sobel_s, t_patch)
#             # t_patch = t_img[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :]
#             # top_left, value = get_matching_pos(s_img, t_patch)
#             similarity[h, w, 0] = value
#             similarity[h, w, 1] = top_left[0]
#             similarity[h, w, 2] = top_left[1]
#     return similarity
#
# if __name__ == '__main__':
#     s_img = cv2.imread('/Users/xiaohan/Desktop/images/shanghai3.tif')
#     t_img = cv2.imread('/Users/xiaohan/Desktop/images/vegas.tif')
#     similarity = compute_patch_similarity(s_img, t_img)
#     print(similarity)
