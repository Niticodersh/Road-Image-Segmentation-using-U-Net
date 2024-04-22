# Visual Computing Lab Mini Project 3

**Student Name:** Nitish Bhardwaj  
**Student ID:** B21AI056  

## Problem Statement

Design a convolutional neural network for performing segmentation tasks on road images using the Cityscapes Dataset, while transforming the masks to a definite number of classes of color encoding.

## Dataset Details

### Regular Dataset
- **Source:** Cityscapes dataset consisting of actual road images along with their corresponding masks. The dataset includes masks in RGB format, and the number of segmented classes is not fixed. Therefore, color encoding is being done using a k-means classifier. Now the number of classes is 12.

### Modifications to Dataset
- **Handling Changes:** In order to handle changes in image capturing due to vehicular movement, random rotation in the range of 0-25 degrees was applied to dataset images and masks by the same amount. Gaussian noise of noise level 0.1 (on a scale of 1) was added to images.
- **Libraries Used:** Changes were made to the dataset using two different libraries, namely Albumentations and OpenCV. Even with applying the same changes, the output images from these two libraries were quite different; hence, datasets made using both libraries were used for training the new model.

#### Modified Dataset 1
- **Changes Applied Using Albumentation Library:** Noise can be seen as light red dots spread across the image. Image and mask rotations covered the whole frame.

#### Modified Dataset 2
- **Changes Applied Using OpenCV Library:** Noise can be seen as light bluish dots spread across the image. Images and masks rotated, leaving black portions in the corners of the frames.
- **Note:** This dataset's validation set is shared with the instructor as mentioned in the instructions.

## Accuracies and Strategies Reports

### Model Training and Validation
- **Regular Model** trained on the regular dataset, **Modified Model 1** trained on Modified Dataset 1, and **Modified Model 2** trained on Modified Dataset 2.
- **Validation IOU Accuracies:**
  - Regular model on regular dataset: **47.63%**
  - Regular model on modified dataset 1: **25.68%**
  - Regular model on modified dataset 2: **25.31%**
  - Modified model 1 on regular dataset: **38.49%**
  - Modified model 1 on modified dataset 1: **40.51%**
  - Modified model 2 on regular dataset: **34.14%**
  - Modified model 2 on modified dataset 2: **48.84%**
- **Strategy Used:** For both modified models, each model was trained from scratch on their respective modified datasets, without using learned weights of previous models, for the same number of 15 epochs as initial. In addition, to increase the performance of the model, the batch size was reduced from 8 to 4 and the SGD optimizer was used instead of Adam.

### Ensemble Model
- An ensemble model of all three models described above was also tested. This model was found more robust to real-life scenarios as seen in the performance chart below.

### Performance Chart

| Model         | Regular Dataset | Modified Dataset 1 | Modified Dataset 2 |
|---------------|-----------------|--------------------|--------------------|
| Regular Model | 47.63%          | 25.68%             | 25.31%             |
| Modified Model 1 | 38.49%      | 40.51%             | 22.12%             |
| Modified Model 2 | 34.14%      | 16.79%             | 48.84%             |
| Ensemble Model   | 45.33%      | 30.51%             | 42.10%             |

**[Access all models and modified datasets here](https://drive.google.com/drive/folders/14g77X0fYFhJOj4Rb2fX9ff2s68Kd20ai?usp=sharing)**


"# Road-Image-Segmentation-using-U-Net" 
