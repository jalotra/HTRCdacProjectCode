## Welcome
If you are here, you have probably read the README.md in the root of this repository. This README adds on to that knowldege.

## Start here
So I have trained models using different techniques on paperspace.com/console/notebooks. They provide free Gpu-s for 6 hrs right <br />
now. Check out them if you want to train or fine-tune this model from scratch. <br />
The model needs <b>5 files </b> to atleast : <br />
1. snapshot-x.data -- Contains the weights for different layers
2. snapshot-x.meta -- Contains the computation graph
3. snapshot.index -- Contains some other things that I don't know yet 
4. charList.txt -- Contains all the different characters that the model can predict.
5. checkpoint -- Contains the name of the latest snapshot at the top to load this particular model in the name "model-checkpoint-path" <br /> 

## What you have to do
1. To get this [repo](https://github.com/jalotra/DoctorFineTune) 
2. I have saved three-different models in [DoctorFineTune/model](https://github.com/jalotra/DoctorFinetune/tree/master/model) that are namely <br />
  a.) <b>doctor_finetunes.zip</b> -- Finetuned the model(c) in this list on custom 512 approx. images that I cropped from prescriptions. <br />
  b.) <b>final_aug_mix_cvl_iam_epochs_91.zip</b> -- Trained model from scratch on IAM+CVL DataBase with augmix dataaugmentation.
      Checkout HTRCdacProjectCode/src/google_augment_and_mix.py and HTRCdacProjectCode/src/augmentations.py for more details.<br />
  c.) <b>final_model_iam_cvl_data_aug_epoch_87.zip</b> -- Trained model from scratch on IAM+CVL Database on normal augmentations among a number 
      of options. Check HTRCdacProjectCode/Training_Code/DataAugmentations.py for details.

## Accuracy on different models 
As defined above these are the char-error-rates and Word-Accuracy 
| MODEL NAME    |CHAR-ERROR-RAT | WORD-ACCURACY  |
| ------------- |:-------------:| -----:|
| a  | 26% | Didn't check |
| b     | 18.52%      |  65 % |
| c | 9.57%     |  73% |

## INSTALLATION 
Clone this [repository](https://github.com/jalotra/DoctorFinetune) and copy the model from a, b or c defined above <br />
in the HTRCdacProjectCode/model and unzip the contents there of the model choosen . <br />
Then <b>run</b> batch_jobs_htr.py after <b>word-segmenting</b> the input images and see the output text in <b>Output_Text</b> Folder. 
