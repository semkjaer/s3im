# Data Descriptor

## Title:
Photoacoustic Data Annotations Supplementing the Paper: "A Study on the Adequacy of Common IQA Measures for Medical Images" [1]

## Authors:
Anna Breger, Janek Gröhl, Thomas Else

## Abstract:
This dataset supplements the study on the adequacy of common image quality assessment (IQA) measures for medical images, specifically focusing on photoacoustic (PA) imaging. The dataset includes 1134 reconstructed images produced by three different reconstruction methods, paired with 378 reference images derived from a photoacoustic imaging system [2]. Algorithm 1 corrects a reconstructed PA image by using the light fluence obtained from simulations. Algorithms 2 and 3 are deep-learning models trained to estimate the absorption coefficient. Expert annotations of image quality are provided for the reconstructed images, with ratings ranging from 1 (very poor) to 4 (very good) based on comparison with the reference images. The annotations were obtained using the publicly available speedyIQA annotation app [3]. This dataset is intended for use in image quality analysis, reconstruction algorithm development, and other research in medical imaging.

## Dataset Description:

### Purpose: 
The primary objective of this dataset is to support research on evaluating the quality of photoacoustic images. The dataset facilitates the comparison of reconstructed images to reference images based on expert ratings, enabling investigations into the adequacy of various reconstruction methods and their effect on image quality.

### Data Collection:
The dataset is based on a previously published open-access dataset containing reconstructed images that estimate the optical absorption coefficient in cross-sectional photoacoustic images of test phantoms. The reconstructed images were generated using three different algorithms. The reference images were acquired using a double-integrating sphere setup, which provided high-quality point estimates for homogeneous material samples. The reconstructed images were then assessed by two independent experts, who rated the overall quality of each image on a scale from 1 to 4 without adjusting contrast or luminance.

### Data Structure:

The dataset consists of two main components:

`Overall_Quality.csv` A CSV file containing expert ratings for the reconstructed images. Each row is formatted as: `filename;rater1;rater2` where filename corresponds to the image file, and rater1 and rater2 are the respective ratings from the two experts.

`normalised_images.zip` A zip folder containing two subdirectories:
- reference/: Contains 378 PNG reference images. The filenames are formatted as imageX.png, where X is the image number.
- algorithms/: Contains 1134 PNG images, representing the outputs of three reconstruction methods. The filenames are formatted as imageX_Y.png, where X corresponds to the reference image number, and Y is the reconstruction method (1, 2, or 3).

## Technical Validation:

To ensure consistent quality assessment, the experts rated the reconstructed images in comparison to the reference images under identical conditions, without altering the contrast or luminance of the images. The reconstructed images were clipped to the reference image’s maximum value to ensure visual comparability. This process simulates a controlled environment for assessing the performance of different reconstruction algorithms in recovering optical absorption properties from photoacoustic measurements.

## Literature:

[1] Anna Breger, Clemens Karner, Ian Selby, Janek Gröhl, Sören Dittmer, Edward Lilley, Judith Babar, Jake Beckford, Timothy J Sadler, Shahab Shahipasand, Arthikkaa Thavakumar, Michael Roberts, Carola-Bibiane Schönlieb: A study on the adequacy of common IQA measures for medical images. Proceedings of MICAD 2024, Manchester, UK ArXiv: https://arxiv.org/abs/2405.19224

[2] Gröhl, J., Else, T.R., Hacker, L., Bunce, E.V., Sweeney, P.W., Bohndiek, S.E.:
Moving beyond simulation: data-driven quantitative photoacoustic imaging using
tissue-mimicking phantoms. IEEE Trans Med Imaging PP (Nov 2023). https://
doi.org/10.1109/TMI.2023.3331198

[3] Selby, I.: Github repository speedyiqa (March 2024), https://github.com/selbs/speedy_iqa 

## License

The data is open access shared under the Creative Commons Attribution 4.0 International license.
Please be so kind as to cite the works [1] and [2], if you want to use the data in your work!