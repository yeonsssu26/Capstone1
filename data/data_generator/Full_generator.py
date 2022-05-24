import os
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import random
import cv2, argparse
import numpy as np
from xml.etree.ElementTree import *
from pascal_voc_writer import Writer


def image_overlay(src, color="#FFFFFF", alpha=0.5):
    overlay = Image.new('RGBA', src.size, color)
    bw_src = ImageEnhance.Color(src).enhance(0.0)
    return Image.blend(bw_src, overlay, alpha)

def insert_black_mask(img):
    black_mask = Image.new('RGBA',img.size, (0, 0, 0))
    val = random.randint(100, 150)
    black_mask.putalpha(val)
    
    img.paste(black_mask, (0, 0,img.size[0], img.size[1]), mask = black_mask)
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
    wpercent = (width_of_plate_holder / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((width_of_plate_holder, hsize), PIL.Image.ANTIALIAS)
        
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    
    
    h_change = abs(car_box_1[1] - car_box_2[1])
    if (abs(car_box_1[0] - car_box_2[0])) < (width_of_plate_holder / 2):
        pts2 = np.float32([[0, h_change], [0, w], [h, 0], [h, w - h_change]])
    elif (abs(car_box_1[0] - car_box_2[0])) >(width_of_plate_holder / 2):
        pts2 = np.float32([[0, 0], [0, w - h_change], [h, h_change], [h, w ]])
    else:
        pts2 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (h, w), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
    img = Image.fromarray(img)
    
    width, height = img.size
    width_prop = car_boxes[0][0]
    height_prop = car_boxes[0][1]
    car_img.paste(img, (width_prop, height_prop, width + width_prop, height + height_prop), mask = img)
    
    bbox = [width_prop, height_prop, width + width_prop, height + height_prop] 
    
    return car_img, bbox


class ImageGenerator:
    def __init__(self, components_path, detection_images_path, 
                 detection_annotations_path, recognition_data_path, cars_path, background_path):
        
        # paths inintialization
        self.components_path = components_path
        self.detection_images_path = detection_images_path
        self.detection_annotations_path = detection_annotations_path
        self.recognition_data_path = recognition_data_path
        self.cars_path = cars_path
        self.background_path = background_path
        
        
        
        # plates initializtion
        plate_files_path = 'plates_type_'
        self.plate_types = ["blank", "1", "2", "3", "4", "5", "6"]
        # plate type 1
        self.Plate_1_images, self.Plate_1_names = self.init_component(
            components_path + plate_files_path + self.plate_types[1] + '/')
        # plate type 2
        self.Plate_2_images, self.Plate_2_names = self.init_component(
            components_path + plate_files_path + self.plate_types[2] + '/')
        # plate type 3
        self.Plate_3_images, self.Plate_3_names = self.init_component(
            components_path + plate_files_path + self.plate_types[3] + '/')
        # plate type 4
        self.Plate_4_images, self.Plate_4_names = self.init_component(
            components_path + plate_files_path + self.plate_types[4] + '/')
        # plate type 5
        self.Plate_5_images, self.Plate_5_names = self.init_component(
            components_path + plate_files_path + self.plate_types[5] + '/')
        # plate type 6
        self.Plate_6_images, self.Plate_6_names = self.init_component(
            components_path + plate_files_path + self.plate_types[6] + '/')

        # components init paths
        chars_path = 'char_'
        nums_path = 'num_'
        regions_path = 'regions_'
        # folders dictionary
        chars_dic = {"black":"black", "white":"white"} 
        nums_dic = {"white":"white", "black":"black"}
        regions_dic = {"region_black":"black", "region_white":"white", "regions_type_4":"type_4"}
        
        # black chars
        self.black_chars_images, self.black_chars_names = self.init_component(
            components_path + chars_path + chars_dic["black"] + '/')
        # white chars
        self.white_chars_images, self.white_chars_names = self.init_component(
            components_path + chars_path + chars_dic["white"] + '/')
        
        # white numbers
        self.white_numbers_images, self.white_numbers_names = self.init_component(
            components_path + nums_path + nums_dic["white"] + '/')
        # black numbers
        self.black_numbers_images, self.black_numbers_names = self.init_component(
            components_path + nums_path + nums_dic["black"] + '/')
        
        # white regions
        self.white_regions_images, self.white_regions_names = self.init_component(
            components_path + regions_path + regions_dic["region_white"] + '/')
        # black regions
        self.black_regions_images, self.black_regions_names = self.init_component(
            components_path + regions_path + regions_dic["region_black"] + '/')
        # type 4 regions
        self.type_4_regions_images, self.type_4_regions_names = self.init_component(
            components_path + regions_path + regions_dic["regions_type_4"] + '/')
        
        
        self.type_1_components = {"Plate_images":self.Plate_1_images,
                                  "Plate_resize":(520, 110),
                                  "Chars":[self.black_chars_images, self.black_chars_names],  
                                  "Numbers":[self.black_numbers_images, self.black_numbers_names], 
                                  "Order":[{"type":"Numbers", "place":(50,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(105,15), "resize":(50,80)}, 
                                           {"type":"Chars", "place":(160,20), "resize":(50,70)},  
                                           {"type":"Numbers", "place":(255,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(310,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(365,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(420,15), "resize":(50,80)}]}
        
        self.type_2_components = {"Plate_images":self.Plate_2_images,
                                  "Plate_resize":(335, 155),
                                  "Chars":[self.black_chars_images, self.black_chars_names],  
                                  "Numbers":[self.black_numbers_images, self.black_numbers_names], 
                                  "Order":[{"type":"Numbers", "place":(10,50), "resize":(40,80)}, 
                                           {"type":"Numbers", "place":(55,50), "resize":(40,80)}, 
                                           {"type":"Chars", "place":(100,60), "resize":(40,70)},  
                                           {"type":"Numbers", "place":(150,50), "resize":(40,80)}, 
                                           {"type":"Numbers", "place":(195,50), "resize":(40,80)}, 
                                           {"type":"Numbers", "place":(240,50), "resize":(40,80)}, 
                                           {"type":"Numbers", "place":(285,50), "resize":(40,80)}]}
        
        self.type_3_components = {"Plate_images":self.Plate_3_images,
                                  "Plate_resize":(335, 170),
                                  "Chars":[self.black_chars_images, self.black_chars_names],  
                                  "Numbers":[self.black_numbers_images, self.black_numbers_names],
                                  "Regions":[self.black_regions_images, self.black_regions_names],
                                  "Order":[{"type":"Regions", "place":(80,10), "resize":(80,50)}, 
                                           {"type":"Numbers", "place":(180,10), "resize":(30,50)}, 
                                           {"type":"Numbers", "place":(215,10), "resize":(30,50)},  
                                           {"type":"Chars", "place":(15,70), "resize":(60,70)}, 
                                           {"type":"Numbers", "place":(80,70), "resize":(50,90)}, 
                                           {"type":"Numbers", "place":(140,70), "resize":(50,90)}, 
                                           {"type":"Numbers", "place":(200,70), "resize":(50,90)}, 
                                           {"type":"Numbers", "place":(260,70), "resize":(50,90)}]}
        
        
        self.type_4_components = {"Plate_images":self.Plate_4_images,
                                  "Plate_resize":(520, 110),
                                  "Chars":[self.black_chars_images, self.black_chars_names],  
                                  "Numbers":[self.black_numbers_images, self.black_numbers_names],
                                  "Regions":[ self.type_4_regions_images, self.type_4_regions_names],
                                  "Order":[{"type":"Regions", "place":(30,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(90,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(145,15), "resize":(50,80)},  
                                           {"type":"Chars", "place":(200,20), "resize":(50,70)}, 
                                           {"type":"Numbers", "place":(270,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(325,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(380,15), "resize":(50,80)}, 
                                           {"type":"Numbers", "place":(435,15), "resize":(50,80)}]}
        
        self.type_5_components = {"Plate_images":self.Plate_5_images,
                                  "Plate_resize":(440, 220),
                                  "Chars":[self.white_chars_images, self.white_chars_names],  
                                  "Numbers":[self.white_numbers_images, self.white_numbers_names],
                                  "Order":[{"type":"Numbers", "place":(115,15), "resize":(65,55)}, 
                                           {"type":"Numbers", "place":(185,15), "resize":(65,55)},  
                                           {"type":"Chars", "place":(255,10), "resize":(65,55)}, 
                                           {"type":"Numbers", "place":(15,80), "resize":(95,125)}, 
                                           {"type":"Numbers", "place":(120,80), "resize":(95,125)}, 
                                           {"type":"Numbers", "place":(225,80), "resize":(95,125)}, 
                                           {"type":"Numbers", "place":(330,80), "resize":(95,125)}]}
        
        self.type_6_components = {"Plate_images":self.Plate_6_images,
                                  "Plate_resize":(335, 170),
                                  "Chars":[self.white_chars_images, self.white_chars_names],  
                                  "Numbers":[self.white_numbers_images, self.white_numbers_names],
                                  "Regions":[self.white_regions_images, self.white_regions_names],
                                  "Order":[
                                           {"type":"Regions", "place":(90,10), "resize":(80,50)}, 
                                           {"type":"Numbers", "place":(170,10), "resize":(35,50)}, 
                                           {"type":"Numbers", "place":(210,10), "resize":(35,50)},  
                                           {"type":"Chars", "place":(15,65), "resize":(60,65)}, 
                                           {"type":"Numbers", "place":(95,65), "resize":(50,95)}, 
                                           {"type":"Numbers", "place":(150,65), "resize":(50,95)}, 
                                           {"type":"Numbers", "place":(205,65), "resize":(50,95)}, 
                                           {"type":"Numbers", "place":(260,65), "resize":(50,95)}]}
        
        
        
        self.components_to_types = {"1":self.type_1_components, 
                                    "2":self.type_2_components, 
                                    "3":self.type_3_components, 
                                    "4":self.type_4_components, 
                                    "5":self.type_5_components,  
                                    "6":self.type_6_components}
        
        self.cars_images, self.cars_list, self.cars_boxes_dic = self.init_car_data(cars_path)
        self.backgrounds, self.background_list = self.init_component(background_path)
             
          
    def init_component(self, files_dir):
        # a bit initialization
        components = list()
        components_list = list()
        files_path = os.listdir(files_dir)
        files_path = [f for f in os.listdir(files_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        for file in files_path:
            component_path = files_dir + file
            component = Image.open(component_path)
            components.append(component)
            components_list.append(file)
        return components, components_list
    
    def build_data(self, desired_number, desired_types, desired_dataset, count):
        
        for iteration in range(desired_number):
            
            # generate desired plate
            plate_type = random.choice(self.plate_types)
            while plate_type not in desired_types:
                plate_type = random.choice(self.plate_types)
            
            generated_plate, label = self.build_plate(plate_type)
            if desired_dataset == 'recognition':
                desired_height = 110
                
                if random.random() < 0.9:
                    generated_plate = image_filtering(generated_plate)
                self.save_plate(generated_plate, label, desired_height)    
                try:
                    
                    print(str(count) + " / " + str(desired_number))
                    count += 1
                
                except:
                    print("Fuck! Save error, probably same combination")
                
                continue
            
               
            full_img, bbox = self.generate_full_image(desired_dataset, generated_plate)
            
            try:
                self.save_img_annotation(full_img, bbox, label, plate_type, count)
                print(str(count) + " / " + str(desired_number)) 
                count += 1
            except:
                print('Could not create or save')
    
    def build_plate(self, plate_type):
        generated_plate = None
        components = self.components_to_types[plate_type]
        plates = components["Plate_images"]
        # choose plate
        Plate_img = random.choice(plates)
        Plate_img = Plate_img.resize(components["Plate_resize"], PIL.Image.ANTIALIAS)
        Generated_Plate, Plate_label = self.place_components(Plate_img, components)
            
        Generated_Plate = Generated_Plate.convert('RGB')
        Generated_Plate = random_bright_contrast(Generated_Plate)
        return Generated_Plate, Plate_label
    
    def place_components(self, Plate_img, components):
        label = str()
        for component in components["Order"]:
            index = random.choice(range(len(components[component["type"]][0])))
            component_img = components[component["type"]][0][index]
            
            component_label = components[component["type"]][1][index][:-4]
            resize_ratio = component["resize"]
            resized_component_img = component_img.resize(resize_ratio, PIL.Image.ANTIALIAS)
            xmin,ymin = component["place"]
            xmax, ymax = xmin + resized_component_img.size[0], ymin + resized_component_img.size[1]
            place_holder = (xmin, ymin, xmax, ymax)
            Plate_img.paste(resized_component_img, place_holder, mask = resized_component_img)
            label += component_label
        return Plate_img, label
    
    def save_plate(self, plate, label, desired_height):
        wpercent = (desired_height / float(plate.size[1]))
        hsize = int((float(plate.size[0]) * float(wpercent)))
        plate = plate.resize((hsize, desired_height), PIL.Image.ANTIALIAS)
        
        plate = plate.convert('RGB')
        plate = plate.convert('L')
        
        plate.save(self.recognition_data_path + label + '.jpg')
        
    def init_car_data(self, cars_path):
        cars_xml = [f for f in os.listdir(cars_path) if f.endswith('.xml')]
        cars_imgs, cars_list = self.init_component(cars_path)
        car_boxes_dic = {}
        for car_xml in cars_xml:
            car_boxes = []
            car_box1 = []
            car_box2 = []
            node = parse(cars_path + car_xml)
            elems = node.findall('object')
            
            for item in (['xmin', 'ymin', 'xmax', 'ymax']):
                car_box1.append(int(int(elems[0].find('bndbox').find(item).text)))
                car_box2.append(int(int(elems[1].find('bndbox').find(item).text)))
            car_boxes.append(car_box1)
            car_boxes.append(car_box2)
            car_boxes_dic[car_xml[:-3] + "png"] = car_boxes
        return cars_imgs, cars_list, car_boxes_dic
    def generate_full_image(self, desired_dataset, plate):
         
        if random.random() < 0.3:
            plate = insert_black_mask(plate)
        
        
        full_img, bbox = None, None
        car = random.choice(self.cars_list)
        print(car)
        car_img = Image.open(self.cars_path + car)
        car_boxes = self.cars_boxes_dic[car]
        background = random.choice(self.background_list)
        background_img = Image.open(self.background_path + background)
        
        car_with_plate_img, bbox = image_augmentation(plate, car_img, car_boxes)
        full_img, bbox = self.insert_car_to_background(background_img, car_with_plate_img, desired_dataset, bbox)
        
        return full_img, bbox
        
    def insert_car_to_background(self, background_img, car_with_plate_img, desired_dataset, bbox):
        
        background_img = background_img.resize((1600, 1200), PIL.Image.ANTIALIAS)
        img_width, img_height = background_img.size
        
        background_car_holder_width = None
        
        if desired_dataset == 'parking':
            background_car_holder_width = random.randint(int(img_width * 0.7), int(img_width * 0.8))
            width_prop = random.randint(int(img_width * 0.05), int(img_width * 0.4))
            height_prop = random.randint(int(img_height * 0.1), int(img_height * 0.3))
        elif desired_dataset == 'cctv':
            width_prop = random.randint(int(img_width * 0.1), int(img_width * 0.4))
            height_prop = random.randint(int(img_height * 0.1), int(img_height * 0.25))
            background_car_holder_width = random.randint(int(img_width * 0.3 * height_prop * 4/1200), 
                                                         int(img_width * 0.45 * height_prop * 4/1200))
        else:
            print('Error!')
            return
        
        wpercent = (background_car_holder_width / float(car_with_plate_img.size[0]))
        hsize = int((float(car_with_plate_img.size[1]) * float(wpercent)))
        resized_car_with_plate_img = car_with_plate_img.resize((background_car_holder_width, hsize), PIL.Image.ANTIALIAS)
        
        # take bbox
        bbox[0] = int(bbox[0] * resized_car_with_plate_img.size[0] / car_with_plate_img.size[0])
        bbox[1] = int(bbox[1] * resized_car_with_plate_img.size[1] / car_with_plate_img.size[1])
        bbox[2] = int(bbox[2] * resized_car_with_plate_img.size[0] / car_with_plate_img.size[0])
        bbox[3] = int(bbox[3] * resized_car_with_plate_img.size[1] / car_with_plate_img.size[1])
        
        # insert car in background
        width, height = resized_car_with_plate_img.size
        

        background_img.paste(resized_car_with_plate_img, (width_prop, height_prop, width + width_prop, height + height_prop), 
                             mask = resized_car_with_plate_img)
        background_img = background_img.convert('L')
        
        bbox[0] += width_prop
        bbox[1] += height_prop
        bbox[2] += width_prop
        bbox[3] += height_prop
        
        #background_img.save('test.jpg')
        
        return background_img, bbox
        
    def save_img_annotation(self, full_img, bbox, label, plate_type, count):
        saved = False
        
        
        #self.detection_images_path = detection_images_path
        #self.detection_annotations_path = detection_annotations_path
            
        
        try:
            # annotation file making
            file_name = str(count) + '_P' + plate_type + '_img.jpg'
            img_save_path = self.detection_images_path + file_name
            full_img.save(img_save_path)


            writer = Writer(img_save_path, full_img.size[0], full_img.size[1])
            writer.addObject('P' + plate_type, bbox[0], bbox[1], bbox[2], bbox[3])
            writer.save(self.detection_annotations_path + file_name[:-3] +'xml')
            saved = True
            
        except:
            print("Could not save")
        
        
        return saved

components_path = 'components_2/'
detection_images_path = '../dataset/yolo'
detection_annotations_path = 'detection_annotations/'
recognition_data_path = 'recognition_data/'

# choose right folder for desired type

# for parking -> 
# cars_path = 'cars/'
# background_path = 'background/'

# for cctv -> 
# cars_path = 'cctv_cars/'
# background_path = 'cctv_background/'


cars_path = 'cars/'
background_path = 'background/'


desired_number = 10000
desired_types = ['1', '2', '3', '4', '5', '6']
# cctv parking recognition provide choose one to generate and store things
desired_dataset = 'parking'
count = 1



generator = ImageGenerator(components_path = components_path, 
                          detection_images_path = detection_images_path, 
                          detection_annotations_path = detection_annotations_path, 
                          recognition_data_path = recognition_data_path, 
                           cars_path = cars_path, 
                           background_path = background_path)
        
generator.build_data(desired_number = desired_number, 
                    desired_types = desired_types, 
                    desired_dataset = desired_dataset, 
                     count = count) 