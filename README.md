# Dust-model-dichotomous-performance-analysis
Using a collated dataset of >90,000 dust point source observations from 9 drylands studies from around the world to assess the performance of dust emission models. Here we use the albedo-based dust emission model developed by Chappell and Webb (2016) in Google Earth Engine. The outputs from each DPS location are standardised and analysed in Python to describe the dichotomous coincidence of observed dust emission with model simulation. Discussion topics are developed from the empirical cumulative distribution function (ECDF) of wind shear velocity conditions during observed dust emission days and all days. These ECDF are also produced in Python.

This Github repository contains:
1. gmd_data: Data folder, which includes the following, required to run both Google Earth Engine (Model) and Python (analysis) code:
    -  Dust Point Source (DPS) observation data, including co-ordinates, dates of observation, and 1-degree grid id
    -  Look-up-tables (LUT) required to run Python code
2. GEE_albedo_point_emission_daily.txt: Google Earth Engine code (.txt), which can be either cut and paste into a GEE portal, or found at the link below. 
4. GMD_User_version.py: Python code used to analyse GEE model output results. 

Link to Google Earth Engine code (Google user account and login required):
https://code.earthengine.google.com/a48652d123f7c56dab0bcdf4a3f9abf5

Download the 'gmd_data' file in a locall accessible loction before you begin running the model. 


Running Google Earth Engine Code

To run the Google Earth Engine (GEE) code, either follow the link above, or copy and past the text file into a GEE web portal. 
****Important***
**You will need to import the DPS observation data (locations and dates) as assets before running the code**. 
Export the model outputs into you google drive folder. Save theese files in a locally accessbile drive before running the Python analysis code. 


Python analysis

This code is sude to produce the figures in the manuscript, including data tables, plots and maps. **Maps are created in QGIS using the gridded output data**

You must insert the location of you user workspace, and the locations of imput data

