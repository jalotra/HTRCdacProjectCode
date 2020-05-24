# HTRCdacProjectCode
Hello Anybody seeing this repository from the future. This repo contains the code that can judge the handwriting of doctor specifially. How does it work ? Lets start simple. 

## Problem Statement 
The problem includes handwriting-recognition and that too of doctors. As you might know or guess if you have seen a doctor-written-prescription that doctor writes in an incomprihensible way. Let me state once more <b>"Given a handwritten prescription of a doctor how can we devise an algorithm that can judge what's written all over the precription." </b>

## Approach 
I knew that deep-learning would help here. So that's what I did. This repo is an example of supervised-sequence labelling algorithm. <br />
If you don't know what is a supervised-sequence labelling algorithm, let me restate that for you. <br />
<b> Explanation </b> <br />
Given a sequence x from a input space X and a target z from a target space Z a supervised-sequnce labelling is to map this x from the input space to z to the output z. Lets say we have D(XxZ) as the total space of elements. In a supervised-learning task we have some elements from D that we call the training set and some disjoint set S' from D that we say is the test-set. Now the task is to generalise on the training-set and apply that knowledge to the test-set.

## What's Next 
So far you get that I have used a supervised-learning algorithm, but what exactly is still unknown.
Lets start with unwrapping that out. <br />
We take as an input a tensor of size batch_size x 128 x 32 x 1 <br />
And we apply 7 CONVOLUTION Layers in this order: 
``` 
Conv1_Output == bs x 64 x 16 x 64
COnv2_Output == bs x 32 x 8 x 128
COnv3_Output == bs X 32 X 8 X 256
Conv4_Ouput == bs X 32 X 4 X 512
Conv5_Output ==  bs X 32X 4 X 512
Conv6_Output == bs X 32X 4 X 512
Conv7_Output == bs X 32 X 1 X 512 
```
Now what have we done is to extract the features in the image. Because that's what a CNN is good at doing. The task stills remains is to map these features to actual-words and to backprop the loss. <br />
<i><b> But here is a catch </b></i>
How can we define map something of the size bs x 32 x 1 x 512 to words. Here come the CTC Loss. You can find much more advanced materials on the net. Just google CTC Loss. A brilliant phd-thesis is [here](https://www.cs.toronto.edu/~graves/phd.pdf). You can find everything about CTC from Alex-Graves. <br />
So basically the thing is that we want loss for each character that might be present and Alex has written a Loss that computes loss according to each char for every time-stamp in the output-map and then backprops it. <br />

Next I use two <b> BDLSTM or Bi-directional LSTMS </b> that can take in consideration what's before and also what's after a character to recognise overtime relation between characters. </b>
Next I apply <b>2 BLSTM Layers </b> :
Input tensor-size = bs x 32 x 1 x 512
``` 
Two Lstm Units having 256 units.
Produce an output of size bs x 32 x 512 .
Then I expand on dim-2. To get a tensor of size bs X 32 X 1 X 512
And finally a atrous-conv to get to a tensor of size bs X 32 x len(chars) + 1.
```
The first 32 defines the total-time stamps because I have defined that a word cannot have more than 32 chars and the second len(chars) + 1 for total chars that can be possible. This includes 26 uppercase and 26 lowercase letters and some punctuation marks. To be exact this : `!"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz` .

## DATA
I have used tranfer-learning because training a model from scratch will not be fruitful for what 510 actual doctor images that I had. <br />
So the goes like this. Took word-images from  :
1. 1,15,319 IAM DATABASE 
2. 1,20,166 CVL_DATABASE

Divided the data into train 90% and test 10% and used `levenshtein distance` as the metric to judge results on the testing dataset. Also I applied Data-Augmentation with probability 0.5 among `rotate, shear, distort, dilate, do_nothing`. That increased the data-size further.  

## Results 
I was able to get to `9.26 %` char-error-rate on the testing-dataset using batch-size == 64. And was able to correctly judge `70.2` % of all the words on testing dataset.   

## Fine-Tuning 
The task remains to fine-tune on the cropped images of the doctor-prescriptions. I had about 512 doctor-images and I fine-tuned the model last-layers namely the 2 Rnn-Layers to achieve a `26%` char-error-rate on it.

## How to use this repo:
1. First of all put all your images that you want to judge the model on in `./Input_Images`.
2. Then run `batch_jobs_segment_images.py`. This will do word-level-segmentation and put the images in `./Word_Segmented_Images/{filename}` folder with names `Box{i}`.
3. Next you must have the model in `./model` folder. For that see `./model/README.md`.
4. Finally run `batch_jobs_htr.py`. This will take all images present in `./Word_Segmented_Images` and forward-prop the bix-iamges through the model and keep the result in `./Output_Text/{foldername}`.
 
