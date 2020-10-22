from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageEnhance, ImageOps


class RandAugment:

    def __init__(self):
        self.transforms = [lambda img: ImageOps.autocontrast(img, cutoff=random.choice(range(10, 40))),
                        lambda img: ImageOps.equalize(img),
                        lambda img: transforms.RandomRotation(180)(img),
                        lambda img: ImageOps.solarize(img, threshold = random.choice(range(128))),
                        lambda img: ImageOps.posterize(img, random.choice(range(1, 9))),
                        lambda img: F.adjust_contrast(img, random.random()+1),
                        lambda img: F.adjust_brightness(img, random.random()+1),
                        lambda img: ImageEnhance.Sharpness(img).enhance(1 + random.choice(range(1, 10)) * random.choice([-1, 1])),
                        lambda img: ImageOps.invert(img),
                        lambda img: transforms.ColorJitter(hue=0.5)(img),
                        lambda img: transforms.RandomAffine(0, shear = 45)(img),
                        lambda img: transforms.RandomAffine(0, shear = [0, 0, -45, 45])(img),
                        lambda img: transforms.RandomAffine(0, translate=(.5, 0))(img),
                        lambda img: transforms.RandomAffine(0, translate=(0, .5))(img)]



    def __call__(self, img):
        for i in range(random.choice([1, 2, 3, 4])):
            trns = random.choice(self.transforms)
        return trns(img)
