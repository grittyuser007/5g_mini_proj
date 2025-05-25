# Creating a Custom Dataset for Injury Detection with Age Group and Gender Classification

This guide will help you create a properly annotated dataset for training a custom YOLO model to detect injuries, age groups, and gender.

## 1. Data Collection

Collect diverse images that include:
- People with various types of injuries
- Different age groups
- Both male and female subjects
- Various lighting conditions, backgrounds, and camera angles
- A mix of close-up and distant shots

Recommended sources:
- Medical training databases (ensure proper licensing)
- Public health repositories
- Staged injury simulations (with consent)
- Public domain images

## 2. Data Organization

Organize your images in the following structure:
```
data/
  custom_dataset/
    images/
      train/
        img001.jpg
        img002.jpg
        ...
      val/
        img101.jpg
        img102.jpg
        ...
```

## 3. Setting Up Roboflow

1. Create a free account on [Roboflow](https://roboflow.com/)
2. Create a new project for multi-label object detection
3. Set up the dataset with the following labels:

   **Injury Types:**
   - burn
   - cut
   - bruise
   - fracture
   - sprain
   - head_injury
   - laceration
   - abrasion
   - no_injury

   **Age Groups:**
   - age_0_20
   - age_21_40
   - age_41_60
   - age_61_plus

   **Gender:**
   - male
   - female

## 4. Annotating Images

1. Upload your images to Roboflow
2. For each person in the image:
   - Draw a bounding box around the person
   - Assign an injury label
   - Also assign an age group label
   - Also assign a gender label
   - Each person should have all three types of labels

## 5. Augmentation in Roboflow

Configure these augmentations to increase dataset variety:
- Brightness adjustment (±25%)
- Rotation (±15°)
- Flip (horizontal)
- Blur (slight, 1px)
- Noise (up to 5%)
- Scale (±10%)

## 6. Generating and Exporting Dataset

1. Generate the dataset with a 70/20/10 split for training/validation/testing
2. Export the dataset in YOLOv8 format
3. Download and extract the dataset to your project's `data/custom_dataset` directory
4. Verify the exported dataset uses the same class indices as in your `data.yaml` file

## 7. Verify the data.yaml

Ensure your data.yaml file looks like:
```yaml
path: data/custom_dataset
train: images/train
val: images/val

names:
  # Injury classes
  0: burn
  1: cut
  2: bruise
  3: fracture
  4: sprain
  5: head_injury
  6: laceration
  7: abrasion
  8: no_injury
  
  # Age group classes
  9: age_0_20
  10: age_21_40
  11: age_41_60
  12: age_61_plus
  
  # Gender classes
  13: male
  14: female
```

## 8. Training

Run the training script:
```
python -m src.utils.train_model --epochs 150 --batch-size 16 --img-size 640
```

## 9. Evaluation

After training, evaluate your model on new test images to ensure it correctly identifies:
- The type of injury
- The age group of the person
- The gender of the person

## Recommendations

- Aim for at least 100 images per class
- Ensure balanced representation across all classes
- Include variation in injury severity, age, and gender
- Use high-quality images where possible
- Include some images with multiple people having different injuries
