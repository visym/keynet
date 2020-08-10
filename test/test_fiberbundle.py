import PIL
import numpy as np
import keynet.fiberbundle
import vipy


def show_fiberbundle_simulation():
    """Save a temp image containing the fiber bundle simulation for the image '../demo/owl.jpg'"""
    img_color = np.array(PIL.Image.open('../demo/owl.jpg').resize( (512,512) ))
    img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)
    return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb').show().savetmp()


def show_fiberbundle_simulation_32x32():
    """Save a 32x32 CIFAR-10 like temp image containing the fiber bundle simulation for the image '../demo/owl.jpg'"""
    img_color_large = np.array(PIL.Image.open('../demo/owl.jpg').resize( (512,512), PIL.Image.BICUBIC ))  
    img_sim_large = keynet.fiberbundle.simulation(img_color_large, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=False)
    img_sim = np.array(PIL.Image.fromarray(np.uint8(img_sim_large)).resize( (32,32), PIL.Image.BICUBIC ).resize( (512,512), PIL.Image.NEAREST ))
    return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb').show().savetmp()

