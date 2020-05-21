# Add the dependency modules to path 
import sys
import os
sys.path.insert(0, "Segmentation_Code")
sys.path.insert(0, "Testing_Code")
sys.path.insert(0, "Inference_Code")

import Segmentation_Code.run as run 
import Inference_Code.custom_validator as validate

# Takes in input images from "./Input_Images"
# Word Level Segmetation is done and the output is kept in "./Word_Segmented_Images/Filename" Folder
def segment():
    input_images_folder = "Input_Images"
    input_images = []
    for files in os.listdir(input_images_folder):
        input_images.append(files)
    print(input_images[0])
    run.run(input_images[0])

# Loads the model from the folder "./model"
# Creates a batch of images for some folder in Word_Segmented_Images example ABCD folder
# And forwards propogates it through the model and saves the ouput in the 
# Output_Text/ABCD Folder
def validate_on_folders():




if __name__ == "__main__":
    # input_images_folder = "Input_Images"
    # input_images = []
    # for files in os.listdir(input_images_folder):
    #     input_images.append(files)    
    segment()