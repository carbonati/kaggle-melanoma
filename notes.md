# external data + patient meta features
pretrain on additional images
Use embeddings + additional patient meta features when blending convolutional embedding features

what if a


# ugly duckling
https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155348
motivation to use `patient_id` for multiple instance learning


- remove duplicates
- generate cv folds
- generate image stats
  - normalize by year, site
    - There are some


# ISIC 2019 preprocessing
images are saved RGB rather than the standard BGR
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 2019 vs 2018 vs 2017
https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164910
The 2019 comp data has 25,000 images and it includes the 12,500 images from 2018 and 2017 comp data. To see which images were in 2018 2017, use the train.csv included within dataset.

Any image where height=450 and width=600 is from 2018 comp, there are 10015 of these images.
(This is also called HAM10000 dataset).
Any image not in 2018 and width!=1024 and height!=1024 is more or less from 2017 comp (This is MSK & UDA-dataset(s) from the ISIC-archive).
Lastly any image with width=1024 and height=1024 was new in 2019. (These are subset of BCN20000 dataset described here)


# Image normalization
- stratify by
  - anatom_site_general_challenge
    - for null `anatom_site_general_challenge`
    - use the full dataset to normalize
    - OR predict the anat then use that to normalize the null site
  - sex
  - data source (2020, 2019, 2018, etc.)


# CV fold generation
  - 10 fold
  - group by
    - patient_id
      - no patient should exist in both the train/val sets
      - some patients with have a positive rate > 0
        - ideally each fold will have an even distribution of malignant rates across patients
          - discretize the number of malignant samples per patient (or the malignant rate)
  - stratify by  
    - target
    - sex
    - anatom_site_general_challenge
    - maybe age (binned/binarized)
  - to build a consistent CV score we can try the following when using external data
    - include external data  in the CV generation
    - only use external data in the train set


# Batch stratification
- enforce a malignant sample is always present in a batch
- what about forcing multiple samples per patient in a batch
  -idk if this makes sense


# Remove hair/noise
https://www.sciencedirect.com/science/article/abs/pii/S0010482597000206




- to do
  - melanoma
    - generate image stats for preprocessed images
    - sync to s3 and on gpu server
    - begin testing ml pipeline to train on a fold of the 224x224 images
    - set up multi-gpu server with vast
    - compare image statistics across different sizes/interpolation algoss

  - car insurance
    - continue working on similarity scoring  


- create new MelanomaDataset
- create new MelanomaModel
- update trainer
    - support multiple metrics  



- trainer
  - should be saving a df_scores table for each session

concat adaptive + gem pooling layers


features to add
  - table with summary/metrics recorded on each epoch
    - requires optimal threshold function
  - clean up model directory functions
  - class weighting
  - postprocess function to generate summary table of each model in each experiment


# experimemnts
- image normalization
  - stratify by anatom_site_general_challenge
- image size



# to do 7/30
  - analysis on roc auc
    - optimal thresholding
  - update the summary report table with more metrics after thresholding
    - best auc step vs best loss step - how far are they? what are the deltas in auc?
  - begin experiments listed above
  - sample weighting stratified by anatom_site_general_challenge
  - incorperate meta features
  - incorperate external data
  - investigate "power averaging" https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165653

  - posthoc analysis on which images are incorrectly being predicted
    - TSNE/UMAP
    - relabeling/removing
  - more architecture searches (after establishing benchmark with b0)
    - B6 looks good?



come back to the 2017 data that is missing anatom site
- come back for malignant imagess



# to do 8/1
  - incorperate external data
  - analysis on roc auc
    - optimal thresholding
  - update the summary report table with more metrics after thresholding
    - best auc step vs best loss step - how far are they? what are the deltas in auc?
  - sample weighting stratified by anatom_site_general_challenge
  - incorperate meta features
  - investigate "power averaging" https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165653
  - posthoc analysis on which images are incorrectly being predicted
    - TSNE/UMAP
    - relabeling/removing
  - more architecture searches (after establishing benchmark with b0)
    - B6 looks good?
  - fix apex problem

2019 is a superset of 2018



# increase resolution on images with borders
image_ids:
  - ISIC_0067913
  - ISIC_0067154

# strange noise in images
- ISIC_0000999_downsampled


# stratification
- i want each fold to have the same "mean" value for a feature


# How to check if model is "stable"
https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/168253
One way to do this is change the seed and run your experiments multiple times and check your cross validation and public leader board score, if they don't differ that much, your model is stable and hopefully you will not recieve that disturbing surprise in the privite leaderboard when the competition ends.


pooling exp

it's very important that i add the EVAL scores to the summary tables..
