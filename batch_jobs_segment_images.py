# Add the dependency modules to path 
import sys
import os
sys.path.insert(0, "Segmentation_Code")
sys.path.insert(0, "Testing_Code")
sys.path.insert(0, "Inference_Code")

import Segmentation_Code.run as run 
import concurrent.futures

# Takes in input images from "./Input_Images"
# Word Level Segmetation is done and the output is kept in "./Word_Segmented_Images/Filename" Folder
def list_images():
    input_images_folder = "Input_Images"
    input_images = []
    for files in os.listdir(input_images_folder):
        input_images.append(files)
    # print(input_images[0])
    return input_images

def  parallel_segment():
    images_list = list_images()
    print(images_list)
    # Create parallelly "PARALLEL" Number of Nodes
    # PARALLEL_SEGMENT_NODES = os.environ.get("PARALLEL_SEGMENT_NODES") 
    # PARALLEL_SEGMENT_NODES = 
    # cnt = 0
    # while(cnt <= len(images_list) - PARALLEL_SEGMENT_NODES):
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         processes = [executor.submit(run.run(images_list[cnt + i])) for i in range(PARALLEL_SEGMENT_NODES)]
    #     cnt += PARALLEL_SEGMENT_NODES
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run.run, images_list)

def create_useful_directories():
    if not os.path.exists("Output_Text/"):
        os.mkdir("Output_Text")
    if not os.path.exists("Word_Segmented_Images/"):
        os.mkdir("Word_Segmented_Images")

if __name__ == "__main__":
    create_useful_directories()
    # input_images_folder = "Input_Images"
    # input_images = []
    # for files in os.listdir(input_images_folder):
    #     input_images.append(files)    
    parallel_segment()