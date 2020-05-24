import sys
import os
sys.path.insert(0, "Segmentation_Code")
sys.path.insert(0, "Testing_Code")
sys.path.insert(0, "Inference_Code")

import Inference_Code.custom_validator as validate
import concurrent.futures

# List folders in "./Word_Segmented_Images"
def list_folders(image_folder = "Word_Segmented_Images"):
    folder_names = []
    for folder in os.listdir(image_folder):
        if os.path.isdir(f"{image_folder}/{folder}"):
            folder_names.append(folder)
    
    return folder_names

# Now for each image in that folder create a batch of size 64 or less 
# And run the custom validator on that image 

def run_custom_validator(image_folder):
    validate.main(image_folder)

def parallel_recognise():
    folders = list_folders()
    folders = [f"Word_Segmented_Images/{x}" for x in folders]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(validate.main, folders)
    # print(*results)
    # Lets write the results to appropriate folders --> Output_Text/folder_name
    results = list(results)
    print(results)
    for i in range(len(results)):
        print(results[0])
        f = open(f"Output_Text/{results[0]}", "a")
        f.write(results[1] + "\n")
        f.close()


# Loads the model from the folder "./model"
# Creates a batch of images for some folder in Word_Segmented_Images example ABCD folder
# And forwards propogates it through the model and saves the ouput in the 
# Output_Text/ABCD Folder
def validate_on_folders():
    pass

if __name__ == "__main__":
    fldrs = list_folders()
    for folder in fldrs:
        print(folder)
    
    # run_custom_validator(f"Word_Segmented_Images/{fldrs[0]}")
    parallel_recognise()