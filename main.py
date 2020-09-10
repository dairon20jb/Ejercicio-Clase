from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
if __name__ == '__main__':
    path = 'C:\PRUEBA'
    image_name = 'dog.png'
    image_namet = 'einstein.png'
    path_file = os.path.join(path, image_name)
    path_filet = os.path.join(path, image_namet)


    mage = cv2.imread(path_file)
    image = cv2.resize(mage, (int(500), int(500)))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)


    maget =  cv2.imread(path_filet)
    imaget = cv2.resize(maget, (int(500), int(500)))
    image_grayt = cv2.cvtColor(imaget, cv2.COLOR_BGR2GRAY)
    image_gray_fftt = np.fft.fft2(image_grayt)
    image_gray_fft_shiftt = np.fft.fftshift(image_gray_fftt)





    image_gray_fft_magt = np.absolute(image_gray_fft_shiftt)
    image_fft_viewt = np.log(image_gray_fft_magt + np.finfo(np.float32).eps)
    image_fft_viewt = image_fft_viewt / np.max(image_fft_viewt)

    image_gray_fft_mag = np.absolute(image_gray_fft_shiftt)
    image_fft_viewt = np.log(image_gray_fft_magt + np.finfo(np.float32).eps)
    image_fft_viewt = image_fft_viewt / np.max(image_fft_viewt)








    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + np.finfo(np.float32).eps)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + np.finfo(np.float32).eps)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # create a low pass filter mask
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    low_pass_mask = np.zeros_like(image_gray)

    freq_cut_off = 0.9 # it should less than 1
    half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
    radius_cut_off = int(freq_cut_off * half_size)
    idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2)  < radius_cut_off
    low_pass_mask[idx] = 1



    num_rowst, num_colst = (image_grayt.shape[0], image_grayt.shape[1])
    enum_rowst = np.linspace(0, num_rowst - 1, num_rowst)
    enum_colst = np.linspace(0, num_colst - 1, num_colst)
    col_itert, row_itert = np.meshgrid(enum_colst, enum_rowst)
    high_pass_mask = np.zeros_like(image_grayt)

    freq_cut_offt = 0.08  # it should less than 1
    half_sizet = num_rowst / 2 - 1  # here we assume num_rows = num_columns
    radius_cut_offt = int(freq_cut_offt * half_sizet)
    idx = np.sqrt((col_itert - half_sizet) ** 2 + (row_itert - half_sizet) ** 2) > radius_cut_offt
    high_pass_mask[idx] = 1

    # filtering via FFT
    fft_filtered = image_gray_fft_shift * low_pass_mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    fft_filteredt = image_gray_fft_shiftt * high_pass_mask
    image_filteredt = np.fft.ifft2(np.fft.fftshift(fft_filteredt))
    image_filteredt = np.absolute(image_filteredt)
    image_filteredt /= np.max(image_filteredt)

    imagecrack= cv2.add(image_filtered,image_filteredt)

    cv2.imshow("Image", imagecrack)
    cv2.waitKey(0)






