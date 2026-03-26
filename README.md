# dl_lab1

## Data

The repository includes an `images` folder that contains the visual data used in the lab. Inside `images` there is a `data` directory where all image files are stored. Also in the `images` folder is the `MAMe_dataset.csv` file which contains metadata and annotations for the specific images (for example: file names, labels, splits, and other attributes). Use `images/data/` to access the image files and `images/MAMe_dataset.csv` to load the related information.

This structure allows code to load image pixels from `images/data/` and join them with labels/attributes from `images/MAMe_dataset.csv` when preparing datasets for training and evaluation.

Note: the VGG models follow ImageNet conventions — they use stacked 3x3 convolutions (stride 1) and typically expect 224×224 inputs. The scripts in `implementations/` use 224×224 by default.