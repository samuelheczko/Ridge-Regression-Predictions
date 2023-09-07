# Holisitc connectome-behaviour analysis tool
This repositry containts source code for the ridge regressio part of the paper: ''. It can be used to build connectomes from fMRI data and to use corss-validated regression models to analyse brain-behavoiur relationsips. 

Required libraries:
1. nilearn
2. scikit-learn
3. scipy
4. numpy
5. nibabel
6. pandas
7. seaborn (for plotting)

Datasets used Amsterdam open MRI data Collection (AOMIC) (1) and Human Connectome project (HCP) (1) Young adults cohort. 

The software is very flexible and can use connections from elsewhere, as in the paper we used HCP networks. Then you have to be careful about the file structure (below) and naming conventions for the connectome file. However, it can also use the subject fMRIs and construct the connectomes. 

To run the regression models the target variables (what are we predicting) need to be loaded. There is no universal convetion to the filetype but in our file stucture regressor files should be in the the subfolder of the source data.

The algorithm produces a result.txt with a name indicating whether feature selection was used (y/nFS) and if so, how many features were used, for the connectome parameterisation method (tangent or Pearson), the size of the training set and the train/test ratio, and the ROI selection method (name of the atlas or data-driven method). Example: ridge_results_yFS_n_feat_2250_both_final_tangent_atlas-DictLearn100_fold_size_200_0.5.csv. 

In this file contains following data for each permutation round: 1
1. GCA_r2/corr/var data of the performace of the main regression model (r2, corr and var of pred vs real)
2. the bias towards education (r2 and corr)
3. perfomance of the model when eucaiton is regressed out from the resutls (r2 and corr), 
4. the perfomace (r2 and corr) of the model using on eductaion as the only feature
5. the performance (r2 and corr) of a simple avarage model where the two predicitons are aveargead. 
6. parameters found during the corss validaion process. 

A similarly named file containing the predictions from the regression model is created with the same information in the title, but called _preds_ instead of _results_. Here each row is either the predicted or real iq value for the subject in the specific permutation.

The software includes two core .py files that define functions to do the connectome construction and regression - connectome.py and regression.py. These scripts are located in the /src folder. Using these scripts, we analyse and create new data files using wrapper scripts which are in the same folder. The wrapper scripts are also used to define the parameters of the analysis.

To execute the scripts we used the slurm protocol. The shell files for this are in the /tools directory.

There is also a Jupyter notebook Notebook_main.ipynb which shows how to go from pre-processed fMRI data in .nii format to cognitive predictions using the code.

To use the code:
1. define the location of the data file (marked as a path at the top of the wrapper scripts).
2. Make sure the data file is structured according to the sample data folder provided in the repository.
3. Make sure to change the paths imgs_paths (subject fMRIs) and atlses (in .nii.gz 3D format) in the connectome extractor, as in the report they follow the specific data structures we used in the paper.

Start by going thought the main_notebook.ipnyb. Note that this is an example file which only takes a subset of subjects into account to make it possible to run locally. The full wrapper scripts are included as .py files, but essentially contain same code as the notebook but for the whole dataset.


The file structure within the data folder we used and the software follows - change as required.

├── data
│   ├── atlases
│   │   ├── 300_ROI_Set
│   │   │   └── label
│   │   ├── HCP
│   │   │   └── HCP
│   │   ├── lawrance2021
│   │   │   ├── label
│   │   │   │   └── Human
│   │   │   │       ├── Anatomical-labels-csv
│   │   │   │       ├── ICBM
│   │   │   │       └── Metadata-json
│   │   │   ├── mask
│   │   │   ├── neuroparc
│   │   │   │   ├── atlases
│   │   │   │   │   ├── label
│   │   │   │   │   │   └── Human
│   │   │   │   │   │       ├── Anatomical-labels-csv
│   │   │   │   │   │       └── Metadata-json
│   │   │   │   │   ├── mask
│   │   │   │   │   ├── reference_brains
│   │   │   │   │   └── transforms
│   │   │   │   └── scripts
│   │   │   ├── reference_brains
│   │   │   └── transforms
│   │   └── templates
│   ├── func_images
│   │   └── AOMIC
│   │       ├── prep_nifti
│   │       └── regressors
│   ├── manual_folds
│   └── results
│       ├── connectomes
│       │   ├── HCP
│       │   │   └── regressors
│       │   ├── pearson
│       │   ├── tangent
│       │   └── examples
│       ├── data_driven_parcellations
│       │   ├── DictMap
│       │   └── ICA
│       ├── feature_importance
│       │   ├── pearson
│       │   └── tangent
│       ├── plots
│       ├── regression_results
│       │   ├── PLS
│       │   │   ├── pearson
│       │   │   └── tangent
│       │   ├── ridge_regression
│       │   │   ├── partial
│       │   │   ├── pearson
│       │   │   │   └── hcp
│       │   │   └── tangent
│       │   ├── examples
│       │   └── SV_regression
│       │       ├── pearson
│       │       └── tangent
│       └── time_series


The data not generated by the algorithm: AOMIC/prep_nifi (_) contains the .nii processed subject images from AOMIC (1) and results/connectomes/hcp/ the netmats2.txt from HCP (2). Both folders also have sub-folders containing the regressors such as IQ and educational level. The atlases /lawrance2021/ contains the neuroparc (3) team atlas collection with metadata (and the ICBM space transform) and 300_ROI_set contains the atlas (4). For the analysis in the paper, we used a reduced subset of the atlases. 

The reuslts folder contains all the data from the algorithm except the hcp connectome, which is netmats2.txt, from which we selected 360 unrelated subjects.

The algorithm produces a result.txt with a name indicating whether feature selection was used (y/nFS) and if so, how many features were used, for the connectome parameterisation method (tangent or Pearson), the size of the training set and the train/test ratio, and the ROI selection method (name of the atlas or data-driven method). Example: ridge_results_yFS_n_feat_2250_both_final_tangent_atlas-DictLearn100_fold_size_200_0.5.csv

In addition to that the software allows for using predefined folds like used in the paper to ensure that the results are comparable.

A similarly named file containing the predictions from the regression model is created with the same information in the title, but called _preds_ instead of _results_. 

1. Snoek, L., van der Miesen, M.M., Beemsterboer, T. et al. The Amsterdam Open MRI Collection, a set of multimodal MRI datasets for individual difference analyses. Sci Data 8, 85 (2021). https://doi.org/10.1038/s41597-021-00870-6
2. David C. Van Essen, Stephen M. Smith, Deanna M. Barch, Timothy E.J. Behrens, Essa Yacoub, Kamil Ugurbil, for the WU-Minn HCP Consortium. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage 80(2013):62-79. 
3. Lawrence, R.M., Bridgeford, E.W., Myers, P.E. et al. Standardizing human brain parcellations. Sci Data 8, 78 (2021). https://doi.org/10.1038/s41597-021-00849-3
4. Benjamin A. Seitzman, Caterina Gratton, Scott Marek, Ryan V. Raut, Nico U.F. Dosenbach, Bradley L. Schlaggar, Steven E. Petersen, Deanna J. Greene, A set of functionally-defined brain regions with improved representation of the subcortex and cerebellum, NeuroImage, Volume 206, 2020, 116290, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2019.116290.



