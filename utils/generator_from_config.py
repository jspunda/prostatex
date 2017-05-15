from keras.preprocessing.image import ImageDataGenerator
import configparser
import os.path

def get_generator(configuration='DEFAULT'):
    
    config_file = open(os.path.join(os.path.dirname(__file__), 'augmentation.ini'))
    
    config = configparser.ConfigParser()
    config.read_file(config_file)
    
    settings = config[configuration]
    
    generator = ImageDataGenerator(featurewise_center=settings.getboolean('featurewise_center'),
    samplewise_center=settings.getboolean('samplewise_center'),
    featurewise_std_normalization=settings.getboolean('featurewise_std_normalization'),
    samplewise_std_normalization=settings.getboolean('samplewise_std_normalization'),
    rotation_range=settings.getint('rotation_range'),
    width_shift_range=settings.getfloat('width_shift_range'),
    height_shift_range=settings.getfloat('height_shift_range'),
    shear_range=settings.getfloat('shear_range'),
    zoom_range=settings.getfloat('zoom_range'),
    channel_shift_range=settings.getfloat('channel_shift_range'),
    fill_mode=settings['fill_mode'],
    horizontal_flip=settings.getboolean('horizontal_flip'),
    vertical_flip=settings.getboolean('vertical_flip'))
    
    return generator