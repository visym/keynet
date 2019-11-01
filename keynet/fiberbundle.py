import scipy
import numpy as np
import math
import PIL
import PIL.ImageStat

def camera_noise(irrad_photons, q_e=.5, sensitivity=0, s_pixel=0,
                     dark_noise=2, bit_depth=12, baseline=90,
                     rs=np.random.RandomState(seed=3)):
    """Function to calculate FPA noise  (sensitivity is same for all pixels for now, no Bayer code yet)
    Camera numbers for q_e, etc. below roughly typical for a good CMOS sensor"""

    # Calculate shot noise 
    photons = rs.poisson(irrad_photons, size=irrad_photons.shape)
    
    # Convert photons to electrons
    electrons = q_e * photons
    
    # Calculate dark noise
    electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons
    
    # Convert to ADU and add baseline ADU value to prevent negative values
    max_ADU     = np.int(2**bit_depth - 1)
    ADU         = (electrons_out * (sensitivity + s_pixel)).astype(np.int) # Convert to integers
    ADU += baseline
    ADU[ADU > max_ADU] = max_ADU # Account for  pixel saturation
    
    return ADU


def simulation(img_color, fiber_core_x=16, fiber_core_y=16, clad_factor=1.25, shear=1, h_xtalk=0.05, v_xtalk=0.05, pad_val=3, image_pixel_size=4, do_camera_noise=True):

    #Loop thru RGB
    for color_num in range(3):
        img = img_color[:,:,color_num]
        rows, cols = img.shape

        #print(rows,cols)  # print image size pixels x pixels
        #image_pixel_size = 4  #pixel size
        image_size_rows = rows * image_pixel_size
        image_size_cols = cols * image_pixel_size

        big_mask = np.zeros( (rows,cols) )  #make mask filled with zeros same size as image
        #pad_val = 3   #pad value
        big_mask = np.pad(big_mask, ((pad_val,pad_val),(pad_val,pad_val)), 'constant')  #pad the mask
        img_pad_val = np.pad(img, ((pad_val,pad_val),(pad_val,pad_val)), 'constant')  #pad the image
        bm_rows, bm_cols = big_mask.shape

        #define fiber bundle parameters
        #fiber_core_x = 16  #core size
        #fiber_core_y = 16

        #clad_factor = 1.25 # set open area to cladding ratio, sets borders on each bundle cell
        clad_size_x = fiber_core_x * clad_factor
        clad_size_y = fiber_core_y * clad_factor
        clad_blocks_rows = math.ceil(image_size_rows / clad_size_x)
        clad_blocks_cols = math.ceil(image_size_cols / clad_size_y)

        #dconvert cladding and fiber core to number of pixels
        clad_pixels_x = math.ceil(clad_size_x / image_pixel_size)
        clad_pixels_y = math.ceil(clad_size_y / image_pixel_size)
        clad_pixels_half_x = math.ceil(clad_pixels_x / 2)
        clad_pixels_half_y = math.ceil(clad_pixels_y / 2)
        fc_pixels_half_x = math.ceil(fiber_core_x / image_pixel_size / 2)
        fc_pixels_half_y = math.ceil(fiber_core_y / image_pixel_size / 2)

        #initialize matrix of centroids for each block of pixels equal to fiber size
        centroid = [[0 for x in range(clad_blocks_rows + pad_val)] for y in range(clad_blocks_cols + pad_val)]

        #shear = 1  #minimum value of 1 here is random from 0-1, maximum is pad_val value (1 = no shear)
        #calculate physical core array center positions and make mask
        for i in range(clad_blocks_rows):
            for j in range(clad_blocks_cols):
                core_pos_pix = [math.ceil(((clad_size_x / 2) + clad_size_x * i) / image_pixel_size + np.random.randint(shear)), math.ceil(((clad_size_y / 2) * ((i + 1) % 2) + clad_size_y * j) / image_pixel_size) + np.random.randint(shear)]
                centroid[i][j] = [core_pos_pix[0],core_pos_pix[1]]
                #print (centroid[i][j][1],core_pos_pix[1])
                for q in range(core_pos_pix[0] - clad_pixels_half_x - 1 , core_pos_pix[0] + clad_pixels_half_x - 1):
                    for x in range(core_pos_pix[1] - clad_pixels_half_y - 1 , core_pos_pix[1] + clad_pixels_half_y - 1):
                        if (np.absolute(core_pos_pix[0] - q) <= (fc_pixels_half_x)) and (np.absolute(core_pos_pix[1] - x) <= (fc_pixels_half_y)):
                            big_mask[q][x] = 1
                            
        big_mask_gray = big_mask < 1   #get mask borders to adjust interstitial gary scale
        big_mask_gray_num = big_mask_gray.astype(int) * 127   #set interstitial gray scale here (0-255)
        sensitivity_pixel = np.zeros((rows + 2*pad_val, cols + 2*pad_val), np.uint8) #initialize sensitivity array
        sensitivity_pixel[4:np.int(rows + pad_val),3:np.int(cols + pad_val)] = 30  #set to 30, just testing function here

        #add code here to read in actual per pixel sensitivity matrix
        #- will need to load a photometric matrix into sensitivty_pixel if that is desired, uniform at 30 right now

        #mask the image with fiber cores and blank interstitial spaces and adjust interstitial gray scale
        fiber_out = np.multiply(img_pad_val,big_mask) + big_mask_gray_num

        #Adjust image so that resolution reflects fiber core size and location on image 
        for i in range(clad_blocks_rows):
            for j in range(clad_blocks_cols):
                row_start = centroid[i][j][0] - clad_pixels_half_x - 1
                row_end =  centroid[i][j][0] + clad_pixels_half_x - 1
                col_start = centroid[i][j][1] - clad_pixels_half_y - 1
                col_end =  centroid[i][j][1] + clad_pixels_half_y - 1
                fiber_out[row_start:row_end - 1 ,col_start:col_end - 1] = np.sum(fiber_out[row_start:row_end - 1,col_start:col_end - 1]) / (clad_pixels_x * clad_pixels_y)

        #adjust image to reflect cross talk from 6 nearest neighbor cells around a central cell
        #can set horitzontal and vertical xtalk coupling separately
        #central cell after xtalk is Ixtalk = Ipre_xtalk*(1-4*vxtalk-2*hxtalk) +(I1+I2+I3+I4)*vxtalk+(I5+I6)*hxtalk

        #image_norm_pre = np.sum(fiber_out[0:clad_blocks_rows,0:clad_blocks_cols])
        image_norm_pre = fiber_out.mean()


        #h_xtalk = .05 #set horizontal xtalk parameter
        #v_xtalk = .05  #set vertical xtalk parameter

        for i in range(clad_blocks_rows-2):
            i = i + 1
            for j in range(clad_blocks_cols-2):
                j = j + 1
                core_1 = fiber_out[centroid[i-1][j-1][0],centroid[i-1][j-1][1]]
                core_2 = fiber_out[centroid[i-1][j+1][0],centroid[i-1][j+1][1]]
                core_3 = fiber_out[centroid[i+1][j-1][0],centroid[i+1][j-1][1]]
                core_4 = fiber_out[centroid[i+1][j+1][0],centroid[i+1][j+1][1]]
                core_5 = fiber_out[centroid[i][j-1][0],centroid[i][j-1][1]]
                core_6 = fiber_out[centroid[i][j+1][0],centroid[i][j+1][1]]
                
                val_weighted = (v_xtalk * (core_1 + core_2 + core_3 + core_4) + h_xtalk * (core_5 + core_6))
                if val_weighted > 255:
                    val_weighted = 255
                cross_talk_fac = (1 - 4 * v_xtalk - 2 * h_xtalk)
                if cross_talk_fac <= 0:
                    cross_talk_fac = 0
                row_start = centroid[i][j][0] - clad_pixels_half_x
                row_end =  centroid[i][j][0] + clad_pixels_half_x
                col_start = centroid[i][j][1] - clad_pixels_half_y
                col_end =  centroid[i][j][1] + clad_pixels_half_y
                fiber_out[row_start:row_end -1 ,col_start:col_end -1] = fiber_out[centroid[i][j][0],centroid[i][j][1]] * cross_talk_fac + val_weighted
        #image_norm_post = np.sum(fiber_out[0:clad_blocks_rows,0:clad_blocks_cols])
        image_norm_post = fiber_out.mean()

        fiber_out = fiber_out * image_norm_pre* 1 / image_norm_post
        fiber_out = np.multiply(fiber_out,big_mask) + big_mask_gray_num
    

        # Calculate FPA noise
        noisy_FPA = (camera_noise(fiber_out, s_pixel = sensitivity_pixel) * 255 / 2**12-1) if do_camera_noise else fiber_out

        if color_num == 0:
            noisy_FPA_red = PIL.Image.fromarray(noisy_FPA)
            noisy_FPA_red_oct = noisy_FPA_red.convert('L')
        if color_num == 1:
            noisy_FPA_green = PIL.Image.fromarray(noisy_FPA )
            noisy_FPA_green_oct = noisy_FPA_green.convert('L')
        if color_num == 2:
            noisy_FPA_blue = PIL.Image.fromarray(noisy_FPA )
            noisy_FPA_blue_oct = noisy_FPA_blue.convert('L')              
        
    img_color_noisy_FPA = PIL.Image.merge("RGB",(noisy_FPA_red_oct,noisy_FPA_green_oct,noisy_FPA_blue_oct))
    return np.array(img_color_noisy_FPA)[pad_val:-pad_val, pad_val:-pad_val, :]


def transform(img_color, outshape=(32,32)):
    assert(len(np.array(img_color).shape) == 3 and np.array(img_color).dtype == np.uint8)
    img_color_large = np.array(PIL.Image.fromarray(np.array(img_color)).resize( (512,512), PIL.Image.NEAREST ))  # nearest neighbor upsample
    img_sim_large = simulation(img_color_large, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=False)
    return PIL.Image.fromarray(np.uint8(img_sim_large)).resize(outshape, PIL.Image.BICUBIC )

