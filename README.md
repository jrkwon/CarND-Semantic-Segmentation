# Semantic Segmentation

### Training

I started with a skeleton provided by the "Project Q&A" session video. Initially, I tried to use my own computer with a GPU. But my GPU doesn't have enough memory to process all the variables in the network. Thus, I had to change my workspace to "Workspace: Semantic Segmentation" provided by Udacity.

I uploaded `data_road` and `vgg` model to the Workspace before testing my `main.py` implementation.

I used `keep_alive.py` to run `main.py` but I had to change one line of the source code and run `keep_alive.py` inside `CardND-Semantic-Segmentation` folder. The Udacity's instructions regarding Workspace  must be fixed. If you followed the Udacity's original instruction, `main.py` cannot find `data` folder.

```
with active_session():
    # do long-running work here
    #call(["python", "CarND-Semantic-Segmentation/main.py"])    
    call(["python", "main.py"])
```

You must be at `/home/workspace/CarND-Semantic-Segmentation` folder to run `keep_alive.py` now. Then run it.

```
python ../keep_alive.py
```

The output from the first trial was terrible. The segmentation results were almost same as the insufficient example output shown in the Udacity's original README file. 

With the hyperparameters below, 0.69 was the final loss which was much higher than it is necessary. 

- l2_regularizer's scale: 1e-3
- keep_prob: 0.5
- learning_rate: 1e-5
- epochs: 20
- batch_size: 4

So I added a `random_normal_initializer` with `stddev` 1e-2 to each layer of the network hoping this addition would reduce the loss.

- l2_regularizer's scale: 1e-3
- random_normal_initializer: 1e-2
- keep_prob: 0.5
- learning_rate: 1e-5
- epochs: 40
- batch_size: 8

This time, the loss was around 0.1 which was much better than the first trial.

![ ](./examples/1545183214.420145/um_000029.png) 
![ ](./examples/1545183214.420145/umm_000072.png) 
![ ](./examples/1545183214.420145/uu_000006.png) 

Then, I increased the epochs to 50 and reduce the batch size to 5. Other hyperparameters were not changed. I had 0.04 for the final loss.

![ ](./examples/1545188284.2499096/um_000029.png) 
![ ](./examples/1545188284.2499096/umm_000072.png) 
![ ](./examples/1545188284.2499096/uu_000006.png) 

At the last several epochs, I did not see much reduction in the loss. So I changed the `learning_rate` and here are examples from my final output.

![ ](./examples/1545192834.8808994/um_000029.png) 
![ ](./examples/1545192834.8808994/umm_000072.png) 
![ ](./examples/1545192834.8808994/uu_000006.png) 

The final hyperparameters that I used here are as follows

- l2_regularizer's scale: 1e-3
- random_normal_initializer: 1e-2
- keep_prob: 0.5
- learning_rate: 9e-4
- epochs: 50
- batch_size: 5


### Results


[![output](https://img.youtube.com/vi/XpjJ4wTMnjg/0.jpg)](https://youtu.be/XpjJ4wTMnjg)

---

The description below is Udacity's original README for this project repository.

----

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Example Outputs
Here are examples of a sufficient vs. insufficient output from a trained network:

Sufficient Result          |  Insufficient Result
:-------------------------:|:-------------------------:
![Sufficient](./examples/sufficient_result.png)  |  ![Insufficient](./examples/insufficient_result.png)

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Why Layer 3, 4 and 7?
In `main.py`, you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

### Optional sections
Within `main.py`, there are a few optional sections you can also choose to implement, but are not required for the project.

1. Train and perform inference on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). Note that the `project_tests.py` is not currently set up to also unit test for this alternate dataset, and `helper.py` will also need alterations, along with changing `num_classes` and `input_shape` in `main.py`. Cityscapes is a much more extensive dataset, with segmentation of 30 different classes (compared to road vs. not road on KITTI) on either 5,000 finely annotated images or 20,000 coarsely annotated images.
2. Add image augmentation. You can use some of the augmentation techniques you may have used on Traffic Sign Classification or Behavioral Cloning, or look into additional methods for more robust training!
3. Apply the trained model to a video. This project only involves performing inference on a set of test images, but you can also try to utilize it on a full video.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
