import os
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import random
import cv2, argparse
import pandas as pd
import numpy as np

# from xml.etree.ElementTree import parse
# from pascal_voc_writer import Writer


def image_overlay(src, color="#FFFFFF", alpha=0.5):
    overlay = Image.new("RGBA", src.size, color)
    bw_src = ImageEnhance.Color(src).enhance(0.0)
    return Image.blend(bw_src, overlay, alpha)


def insert_black_mask(img):
    black_mask = Image.new("RGBA", img.size, (0, 0, 0))
    val = random.randint(100, 150)
    black_mask.putalpha(val)

    img.paste(black_mask, (0, 0, img.size[0], img.size[1]), mask=black_mask)
    return img


def random_bright_contrast(img):
    img = ImageEnhance.Contrast(img)
    num_contrast = random.uniform(0.7, 1.5)
    img = img.enhance(num_contrast)
    num_brightness = random.uniform(0.8, 1.0)
    img = ImageEnhance.Brightness(img)
    img = img.enhance(num_brightness)

    if random.random() < 0.1:
        img = img.filter(ImageFilter.BLUR)

    return img


def image_filtering(img, ang_range=1.2, shear_range=1.5, trans_range=1):

    img = np.array(img)

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 0.9)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    img = Image.fromarray(img)

    return img


def image_augmentation(img, car_img, car_boxes):
    bbox = None
    car_box_1 = car_boxes[0]
    car_box_2 = car_boxes[1]

    width_of_plate_holder = car_box_1[2] - car_box_1[0]
    wpercent = width_of_plate_holder / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((width_of_plate_holder, hsize), PIL.Image.ANTIALIAS)

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])

    h_change = abs(car_box_1[1] - car_box_2[1])
    if (abs(car_box_1[0] - car_box_2[0])) < (width_of_plate_holder / 2):
        pts2 = np.float32([[0, h_change], [0, w], [h, 0], [h, w - h_change]])
    elif (abs(car_box_1[0] - car_box_2[0])) > (width_of_plate_holder / 2):
        pts2 = np.float32([[0, 0], [0, w - h_change], [h, h_change], [h, w]])
    else:
        pts2 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(
        img,
        M,
        (h, w),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0, 0],
    )
    img = Image.fromarray(img)

    width, height = img.size
    width_prop = car_boxes[0][0]
    height_prop = car_boxes[0][1]
    car_img.paste(
        img,
        (width_prop, height_prop, width + width_prop, height + height_prop),
        mask=img,
    )

    bbox = [width_prop, height_prop, width + width_prop, height + height_prop]

    return car_img, bbox


class ImageGenerator:
    def __init__(
        self,
        components_path,
        plate_images_path,
        detection_annotations_path,
        recognition_data_path,
        cars_path,
        background_path,
    ):

        # paths inintialization
        self.components_path = components_path
        self.plate_images_path = plate_images_path
        self.detection_annotations_path = detection_annotations_path
        self.recognition_data_path = recognition_data_path
        self.cars_path = cars_path
        self.background_path = background_path

        # plates initializtion
        plate_files_path = "plates_type_"
        self.plate_types = ["blank", "1", "2", "3", "4", "5", "6"]
        # plate type 1
        self.Plate_1_images, self.Plate_1_names = self.init_component(
            components_path + plate_files_path + self.plate_types[1] + "/"
        )
        # plate type 2
        self.Plate_2_images, self.Plate_2_names = self.init_component(
            components_path + plate_files_path + self.plate_types[2] + "/"
        )
        # plate type 3
        self.Plate_3_images, self.Plate_3_names = self.init_component(
            components_path + plate_files_path + self.plate_types[3] + "/"
        )
        # plate type 4
        self.Plate_4_images, self.Plate_4_names = self.init_component(
            components_path + plate_files_path + self.plate_types[4] + "/"
        )
        # plate type 5
        self.Plate_5_images, self.Plate_5_names = self.init_component(
            components_path + plate_files_path + self.plate_types[5] + "/"
        )
        # plate type 6
        self.Plate_6_images, self.Plate_6_names = self.init_component(
            components_path + plate_files_path + self.plate_types[6] + "/"
        )

        # components init paths
        chars_path = "char_"
        nums_path = "num_"
        regions_path = "regions_"
        # folders dictionary
        chars_dic = {"black": "black", "white": "white"}
        nums_dic = {"white": "white", "black": "black"}
        regions_dic = {
            "region_black": "black",
            "region_white": "white",
            "regions_type_4": "type_4",
        }

        # black chars
        self.black_chars_images, self.black_chars_names = self.init_component(
            components_path + chars_path + chars_dic["black"] + "/"
        )
        # white chars
        self.white_chars_images, self.white_chars_names = self.init_component(
            components_path + chars_path + chars_dic["white"] + "/"
        )

        # white numbers
        self.white_numbers_images, self.white_numbers_names = self.init_component(
            components_path + nums_path + nums_dic["white"] + "/"
        )
        # black numbers
        self.black_numbers_images, self.black_numbers_names = self.init_component(
            components_path + nums_path + nums_dic["black"] + "/"
        )

        # white regions
        self.white_regions_images, self.white_regions_names = self.init_component(
            components_path + regions_path + regions_dic["region_white"] + "/"
        )
        # black regions
        self.black_regions_images, self.black_regions_names = self.init_component(
            components_path + regions_path + regions_dic["region_black"] + "/"
        )
        # type 4 regions
        self.type_4_regions_images, self.type_4_regions_names = self.init_component(
            components_path + regions_path + regions_dic["regions_type_4"] + "/"
        )

        self.type_1_components = {
            "Plate_images": self.Plate_1_images,
            "Plate_resize": (520, 110),
            "Chars": [self.black_chars_images, self.black_chars_names],
            "Numbers": [self.black_numbers_images, self.black_numbers_names],
            "Order": [
                {"type": "Numbers", "place": (50, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (105, 15), "resize": (50, 80)},
                {"type": "Chars", "place": (160, 20), "resize": (50, 70)},
                {"type": "Numbers", "place": (255, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (310, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (365, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (420, 15), "resize": (50, 80)},
            ],
        }

        self.type_2_components = {
            "Plate_images": self.Plate_2_images,
            "Plate_resize": (335, 155),
            "Chars": [self.black_chars_images, self.black_chars_names],
            "Numbers": [self.black_numbers_images, self.black_numbers_names],
            "Order": [
                {"type": "Numbers", "place": (10, 50), "resize": (40, 80)},
                {"type": "Numbers", "place": (55, 50), "resize": (40, 80)},
                {"type": "Chars", "place": (100, 60), "resize": (40, 70)},
                {"type": "Numbers", "place": (150, 50), "resize": (40, 80)},
                {"type": "Numbers", "place": (195, 50), "resize": (40, 80)},
                {"type": "Numbers", "place": (240, 50), "resize": (40, 80)},
                {"type": "Numbers", "place": (285, 50), "resize": (40, 80)},
            ],
        }

        self.type_3_components = {
            "Plate_images": self.Plate_3_images,
            "Plate_resize": (335, 170),
            "Chars": [self.black_chars_images, self.black_chars_names],
            "Numbers": [self.black_numbers_images, self.black_numbers_names],
            "Regions": [self.black_regions_images, self.black_regions_names],
            "Order": [
                {"type": "Regions", "place": (80, 10), "resize": (80, 50)},
                {"type": "Numbers", "place": (180, 10), "resize": (30, 50)},
                {"type": "Numbers", "place": (215, 10), "resize": (30, 50)},
                {"type": "Chars", "place": (15, 70), "resize": (60, 70)},
                {"type": "Numbers", "place": (80, 70), "resize": (50, 90)},
                {"type": "Numbers", "place": (140, 70), "resize": (50, 90)},
                {"type": "Numbers", "place": (200, 70), "resize": (50, 90)},
                {"type": "Numbers", "place": (260, 70), "resize": (50, 90)},
            ],
        }

        self.type_4_components = {
            "Plate_images": self.Plate_4_images,
            "Plate_resize": (520, 110),
            "Chars": [self.black_chars_images, self.black_chars_names],
            "Numbers": [self.black_numbers_images, self.black_numbers_names],
            "Regions": [self.type_4_regions_images, self.type_4_regions_names],
            "Order": [
                {"type": "Regions", "place": (30, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (90, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (145, 15), "resize": (50, 80)},
                {"type": "Chars", "place": (200, 20), "resize": (50, 70)},
                {"type": "Numbers", "place": (270, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (325, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (380, 15), "resize": (50, 80)},
                {"type": "Numbers", "place": (435, 15), "resize": (50, 80)},
            ],
        }

        self.type_5_components = {
            "Plate_images": self.Plate_5_images,
            "Plate_resize": (440, 220),
            "Chars": [self.white_chars_images, self.white_chars_names],
            "Numbers": [self.white_numbers_images, self.white_numbers_names],
            "Order": [
                {"type": "Numbers", "place": (115, 15), "resize": (65, 55)},
                {"type": "Numbers", "place": (185, 15), "resize": (65, 55)},
                {"type": "Chars", "place": (255, 10), "resize": (65, 55)},
                {"type": "Numbers", "place": (15, 80), "resize": (95, 125)},
                {"type": "Numbers", "place": (120, 80), "resize": (95, 125)},
                {"type": "Numbers", "place": (225, 80), "resize": (95, 125)},
                {"type": "Numbers", "place": (330, 80), "resize": (95, 125)},
            ],
        }

        self.type_6_components = {
            "Plate_images": self.Plate_6_images,
            "Plate_resize": (335, 170),
            "Chars": [self.white_chars_images, self.white_chars_names],
            "Numbers": [self.white_numbers_images, self.white_numbers_names],
            "Regions": [self.white_regions_images, self.white_regions_names],
            "Order": [
                {"type": "Regions", "place": (90, 10), "resize": (80, 50)},
                {"type": "Numbers", "place": (170, 10), "resize": (35, 50)},
                {"type": "Numbers", "place": (210, 10), "resize": (35, 50)},
                {"type": "Chars", "place": (15, 65), "resize": (60, 65)},
                {"type": "Numbers", "place": (95, 65), "resize": (50, 95)},
                {"type": "Numbers", "place": (150, 65), "resize": (50, 95)},
                {"type": "Numbers", "place": (205, 65), "resize": (50, 95)},
                {"type": "Numbers", "place": (260, 65), "resize": (50, 95)},
            ],
        }

        self.components_to_types = {
            "1": self.type_1_components,
            "2": self.type_2_components,
            "3": self.type_3_components,
            "4": self.type_4_components,
            "5": self.type_5_components,
            "6": self.type_6_components,
        }

        ####################################################################################
        # self.cars_images, self.cars_list, self.cars_boxes_dic = self.init_car_data(
        #     cars_path
        # )
        # self.backgrounds, self.background_list = self.init_component(background_path)

    def init_component(self, files_dir):
        # a bit initialization
        components = list()
        components_list = list()
        files_path = os.listdir(files_dir)
        files_path = [
            f for f in os.listdir(files_dir) if f.endswith(".jpg") or f.endswith(".png")
        ]

        for file in files_path:
            component_path = files_dir + file
            component = Image.open(component_path)
            components.append(component)
            components_list.append(file)
        return components, components_list

    def build_data(self, desired_number, desired_types, desired_dataset, count):
        # def build_data(self, desired_number, desired_types, count):

        for iteration in range(desired_number):

            # generate desired plate
            plate_type = random.choice(self.plate_types)
            while plate_type not in desired_types:
                plate_type = random.choice(self.plate_types)

            generated_plate, label = self.build_plate(plate_type)

            # parking??
            # if desired_dataset == "recognition":
            #     desired_height = 110

            #     if random.random() < 0.9:
            #         generated_plate = image_filtering(generated_plate)
            #     self.save_plate(generated_plate, label, desired_height)
            #     try:

            #         print(str(count) + " / " + str(desired_number))
            #         count += 1

            #     except:
            #         print("Fuck! Save error, probably same combination")

            #     continue

            ################################################################################
            # full_img, bbox = self.generate_full_image(desired_dataset, generated_plate)

            # try:
            #     # self.save_img_annotation(full_img, bbox, label, plate_type, count)
            #     self.save_img_annotation(generated_plate, label, plate_type, count)
            #     print(str(count) + " / " + str(desired_number))
            #     count += 1

            # except:
            #     print("Could not create or save")

            self.save_img_annotation(generated_plate, label, plate_type, count)
            print(str(count) + " / " + str(desired_number))
            count += 1

    def build_plate(self, plate_type):
        generated_plate = None
        components = self.components_to_types[plate_type]
        plates = components["Plate_images"]
        # choose plate
        Plate_img = random.choice(plates)
        Plate_img = Plate_img.resize(components["Plate_resize"], PIL.Image.ANTIALIAS)
        Generated_Plate, Plate_label = self.place_components(Plate_img, components)

        Generated_Plate = Generated_Plate.convert("RGB")
        Generated_Plate = random_bright_contrast(Generated_Plate)
        return Generated_Plate, Plate_label

    def place_components(self, Plate_img, components):
        label = str()
        for component in components["Order"]:
            index = random.choice(range(len(components[component["type"]][0])))
            component_img = components[component["type"]][0][index]

            component_label = components[component["type"]][1][index][:-4]
            resize_ratio = component["resize"]
            resized_component_img = component_img.resize(
                resize_ratio, PIL.Image.ANTIALIAS
            )
            xmin, ymin = component["place"]
            xmax, ymax = (
                xmin + resized_component_img.size[0],
                ymin + resized_component_img.size[1],
            )
            place_holder = (xmin, ymin, xmax, ymax)
            Plate_img.paste(
                resized_component_img, place_holder, mask=resized_component_img
            )
            label += component_label
        return Plate_img, label

    # def save_img_annotation(self, full_img, bbox, label, plate_type, count):  #
    def save_img_annotation(self, plate_img, label, plate_type, count):  #
        saved = False

        # print("label param : ", label)

        # self.plate_images_path = plate_images_path
        # self.detection_annotations_path = detection_annotations_path

        # try:
        #     # annotation file making
        #     file_name = str(count) + "_P" + plate_type + "_img.jpg"
        #     img_save_path = self.plate_images_path + file_name
        #     plate_img.save(img_save_path)

        #     annotations = []
        #     vectors = self.components_to_types[plate_type]["Order"]
        #     for v in vectors:
        #         x1, y1 = v["place"]
        #         x2 = v["place"][0] + v["resize"][0]
        #         y2 = v["place"][1] + v["resize"][1]
        #         annotations.append([img_save_path, x1, y1, x2, y2])
        #     saved = True

        #     annotations = np.array(annotations)
        #     label = label.split("")
        #     if plate_type in ["4","5","6"]:
        #         col = [label[0] + label[1]] + label[2:]
        #     else:
        #         col = label

        #     print(col)
        #     np.insert(annotations, -1, col, axis=1)

        #     print(annotations)

        #     pd.DataFrame(
        #         annotations, columns=["image path", "x1", "y1", "x2", "y2", "label"]
        #     )
        #     pd.to_csv("annotations.csv", mode="a", header=False, index=True)

        # except:
        #     print("Could not save")

        file_name = str(count) + "_P" + plate_type + "_img.jpg"
        img_save_path = self.plate_images_path + file_name
        plate_img.save(img_save_path)

        annotations = []
        vectors = self.components_to_types[plate_type]["Order"]
        width, height = self.components_to_types[plate_type]["Plate_resize"]
        for v in vectors:
            x1, y1 = v["place"]
            x2 = v["place"][0] + v["resize"][0]
            y2 = v["place"][1] + v["resize"][1]
            annotations.append([file_name, width, height, x1, y1, x2, y2])
        saved = True

        annotations = np.array(annotations)
        label = list(label)
        # print("label : ", label)

        if plate_type in ["3", "4", "6"]:
            col = label[:3] + [label[3] + label[4]] + label[5:]
        else:
            col = label[:2] + [label[2] + label[3]] + label[4:]

        # print(col)
        # np.insert(annotations, 5, col, axis=1)

        # print(annotations)

        df = pd.DataFrame(
            # annotations, columns=["image path", "x1", "y1", "x2", "y2", "label"]
            annotations,
            columns=["image path","width","height", "x1", "y1", "x2", "y2"],
        )
        df["label"] = col

        print(df)

        df.to_csv(self.plate_images_path+"annotations.csv", mode="a", header=False, index=False)

        return saved


components_path = "components_2/"
plate_images_path = "../dataset/test/test1/"
detection_annotations_path = "detection_annotations/"
recognition_data_path = "recognition_data/"

# choose right folder for desired type

# for parking ->
# cars_path = 'cars/'
# background_path = 'background/'

# for cctv ->
# cars_path = 'cctv_cars/'
# background_path = 'cctv_background/'


cars_path = "cars/"
background_path = "background/"


desired_number = 500
desired_types = ["1", "2", "3", "4", "5", "6"]
# cctv parking recognition provide choose one to generate and store things
desired_dataset = "parking"
count = 1


generator = ImageGenerator(
    components_path=components_path,
    plate_images_path=plate_images_path,
    detection_annotations_path=detection_annotations_path,
    recognition_data_path=recognition_data_path,
    cars_path=cars_path,
    background_path=background_path,
)

generator.build_data(
    desired_number=desired_number,
    desired_types=desired_types,
    desired_dataset=desired_dataset,
    count=count,
)
