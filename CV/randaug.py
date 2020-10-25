from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageEnhance, ImageOps


class RandAugment:

    def __init__(self):
        self.transforms = [lambda img: ImageOps.autocontrast(img, cutoff=random.choice(range(10, 40))), #auto contrast
                        lambda img: ImageOps.equalize(img), #equalize
                        lambda img: transforms.RandomRotation(180)(img), #randomly rotate image from -180~180 degrees
                        lambda img: ImageOps.solarize(img, threshold = random.choice(range(128))), #solarize image from random threshold 0~128
                        lambda img: ImageOps.posterize(img, random.choice(range(1, 9))), #posterize image (magnitude 1~8)
                        lambda img: F.adjust_contrast(img, random.random()+1), #adjust constrast by 1~2
                        lambda img: F.adjust_brightness(img, random.random()+1),#adjust brightness by 1~2
                        lambda img: ImageEnhance.Sharpness(img).enhance(1 + random.choice(range(1, 10)) * random.choice([-1, 1])), #sharpen image randomly
                        lambda img: ImageOps.invert(img), #invert image
                        lambda img: transforms.ColorJitter(hue=0.5)(img), #jitter image by -0.5~0.5
                        lambda img: transforms.RandomAffine(0, shear = 45)(img), #randomly shear to x axis -45~45
                        lambda img: transforms.RandomAffine(0, shear = [0, 0, -45, 45])(img),#randomly shear to y axis -45~45
                        lambda img: transforms.RandomAffine(0, translate=(.5, 0))(img), #randomly translate along x -50%~50%
                        lambda img: transforms.RandomAffine(0, translate=(0, .5))(img)] #randomly translate along y -50%~50%



    def __call__(self, img):
        for i in range(random.choice([1, 2, 3, 4])): #sample K=1~4
            trns = random.choice(self.transforms) #transform K times
        return trns(img)
