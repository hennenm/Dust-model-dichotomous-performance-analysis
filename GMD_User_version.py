#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:35:28 2020

@author: markhennen
"""

# Import correct libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

from statistics import mean


# Set working Directory
os.chdir ('# - Set user directory - #')

# Input data folder location
dat_in =    '# - Set directory for GEE results - #/'
# Output data folder location
dat_out =   '# - Set directory to store results - #/'

#===================================================================================================================================
#---------------------------------------Empty lists for loops-----------------------------------------------------------------------

index_list =    []
case_list =     []     
              
cas_stds    = ['Baddock', 'Bullard', 'Eckardt', 'Hennen', 'Kandakji', 'Lee', 'Nobakht', 'Schepanski', 'vonHoldt']
cas_reg     = ['N. America (Baddock)', 'Australia (Bullard)', 'Sthn. Africa (Eckardt)', 'Middle East (Hennen)', 'N. America (Kandakji)', 'N. America (Lee)', 'Central Asia (Nobakht)', 'N. Africa (Schepanski)', 'Sthn. Africa (von Holdt)']
   
cas_reg_di  = dict(zip(cas_stds, cas_reg))

#===================================================================================================================================
#--------------------------------------- Code controls ------------------------------------------------------------------------------
#===================================================================================================================================
# Run batch or just plots
run_batch   = True
# Select which plots to make
cont_table  = True
ecdf        = True

# Specify which columns to import
columns =       ['system:index','date','\ufeffid', 'id', 'usuh', 'fmb', 'us']

# Select which files to run
case_list =       ['Baddock','Schepanski','Hennen', 'Nobakht', 'Bullard', 'Lee', 'Kandakji', 'Eckardt', 'vonHoldt']

# Import dps look up table
dps_lut     = np.load(f'{dat_in}/data/dps_lut.npy', allow_pickle = 'TRUE')  # Import dps_lut.npy from github
us_grid     = pd.read_csv(f'{dat_in}/data/Global_us.csv')                   # Import Global_us.csv from github
us_dict     = dict(zip(us_grid.id, us_grid.Nov21_mean))

ecdf_list   = {k:[] for k in case_list}
sort_list   = {}

#===================================================================================================================================
#------------------------------ Create data for each data point --------------------------------------------------------------------
#===================================================================================================================================
if run_batch == True:
    for entry in os.scandir(dat_in):

        if (entry.path.endswith(".csv")):
            case_path = entry.path.split('/')[-1]
            
            case_std = case_path.split('_')[4]
                           
            # choose which data to input in the loop
            if case_std in case_list:
                # Update the index list to be added to looped dataframe
                index_list.append(case_std)
            else:                   
                continue
            
            #------------------------------- Grid id ------------------------------------------------------------------------------
            dat_coor        = f"{case_std.split('-')[0]}_gid.csv"
            # Read in coordinates data
            Coor            = pd.read_csv(f'{dat_in}{dat_coor}')
            # Clean data column headers
            Coor.columns    = Coor.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            loc_dict1       = dict(zip(Coor.id,Coor.grid_id))
            #----------------------------------------------------------------------------------------------------------------------

            index_group = set(loc_dict1.values())

            # Observation data container
            chunk_group = []
            
            # load active dps data
            active_id = dps_lut[4][case_std]    

            # Grid box date containers
            hit_grp      = {k:[] for k in index_group}          
            nan_grp      = {k:[] for k in index_group}
            
            # grid_count  = Coor.groupby('co_id1')['id'].size().to_dict()
            grid_count       = Coor.groupby('grid_id')['id'].size().to_dict()
               
            for index, Data in enumerate(pd.read_csv(entry.path, usecols = lambda x: x.lower() in columns, chunksize= dps_lut[3][case_std]),start=1):
              
                print(case_std, 'percent complete: ', round(index/dps_lut[2][case_std]*100,0))
                # print(chunk_s)
                
                # Clean all column headings by replacing all gaps and uncapitalising column headings
                Data.columns = Data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
                # Data as type datetime 
                Data['date'] = Data['date'].astype('datetime64[ns]')
                
                # Extracting the simulation date information from system:index column 
                Data.insert(0,'sim_date', Data['system:index'].str[:10])#expand=True)
                Data['sim_date'] = Data['sim_date'].str.replace('_', '/')
                Data['sim_date'] = Data['sim_date'].astype('datetime64[ns]')
                # Data['sim_date'] = pd.to_datetime(Data.sim_date, format='%Y_%m_%d', utc=True)#.dt.date
                Data = Data.drop(['system:index'], axis=1) #remove spare and index columns 
                        
                if '\ufeffid' in list(Data):
                    Data = Data.rename(columns = {'\ufeffid': 'id'})
                
                Data = Data.loc[Data.id.isin(active_id)]

                # Add unique grid co-or id
                Data ['uni_id1'] = Data.id.map(loc_dict1)
                
                obs_chunk = Data[Data.date == Data.sim_date]
                    
                chunk_group.append(obs_chunk)                             

#===================================================================================================================================
#---------------------------------- Dichotomous per chunk  -------------------------------------------------------------------------
#===================================================================================================================================                                     
                # for d in range(0,len(hit_grp)):
                hit_list = {k:v for (k,v) in Data.loc[Data['fmb'] > 0].groupby('uni_id1')['sim_date'].apply(list).items()}
                nan_list = {k:v for (k,v) in Data.loc[Data['fmb'].isnull()].groupby('uni_id1')['sim_date'].apply(list).items()}
                
                hit_sum = {k:v for (k,v) in Data.loc[Data['fmb'] >0].groupby('uni_id1')['us'].sum().items()}
                hit_count = {k:v for (k,v) in Data.loc[Data['fmb'] >0].groupby('uni_id1')['us'].count().items()}
                                       
                for i in hit_list.keys():
                    hit_grp[i].extend(hit_list[i])
 
        
                for i in nan_list.keys():
                    nan_grp[i].extend(nan_list[i])
                            
                if ecdf == True:
                    ecdf_list[case_std]   = np.hstack((ecdf_list[case_std],(Data.loc[Data['fmb'] >= 0].groupby(['uni_id1','sim_date'])['us'].max())))

#============================= End of Case_study chunck loop =======================================================================      
            if ecdf == True:
                np.save(f'{dat_out}tot_cdf_1', ecdf_list)
#===================================================================================================================================
#-------------------------------------- Total Dichotomous --------------------------------------------------------------------------
#===================================================================================================================================                               
            # Pull together all chunked data
            obs_grp = pd.concat(chunk_group) 
                
                
            # Grid box frequency counts
            obs_hit     = {k:[] for k in index_group}
            obs_nan     = {k:[] for k in index_group}
            
            obs_us      = {k:0 for k in index_group}
            obs_count   = {k:0 for k in index_group}
             
            obs_hit_ls  = {k:v for (k,v) in obs_grp.loc[obs_grp['fmb'] >= 0].groupby('uni_id1')['sim_date'].apply(list).items()}
            obs_nan_ls  = {k:v for (k,v) in obs_grp.loc[obs_grp['fmb'].isnull()].groupby('uni_id1')['sim_date'].apply(list).items()}
            
            hit_sum     = {k:v for (k,v) in obs_grp.loc[obs_grp['fmb'] >=0].groupby('uni_id1')['us'].sum().items()}
            hit_count   = {k:v for (k,v) in obs_grp.loc[obs_grp['fmb'] >=0].groupby('uni_id1')['us'].count().items()}
                       
            # Loop through grid boxes to list all hit and nan dates
            for i in obs_hit_ls.keys():
                obs_hit[i].extend(obs_hit_ls[i])
                obs_us[i]    += hit_sum[i]
                obs_count[i] += hit_count[i]
            for i in obs_nan_ls.keys():
                obs_nan[i].extend(obs_nan_ls[i])
            
            # Compile all dates per dichotmous outcome 
            hits        = {k:set(hit_grp[k]).intersection(set(obs_hit[k])) for k in index_group} 
            miss        = {k:np.setdiff1d(obs_hit[k],hit_grp[k]) for k in index_group}
            false       = {k:np.setdiff1d(hit_grp[k],obs_hit[k]) for k in index_group}
            # Determine all nans by reo
            nan         = {k:np.setdiff1d(nan_grp[k],hit_grp[k]) for k in index_group}
            nan         = {k:np.setdiff1d(nan[k],miss[k]) for k in index_group}
            
            # Empty counts for dichotomus outcomes
            hit_c       = 0
            miss_c      = 0
            false_c     = 0
            cor_neg     = 0
            nan_c       = 0
            ops         = 0
            
            # Empty dictionary to store results in loop
            results     = {}
            
            # Loop through each grid cell to calculate results
            for i in index_group:
                if len(hit_grp[i]) > 0:
                    results[i] = [len(hits[i]),len(miss[i]),len(false[i]), 
                                  dps_lut[0][case_std] - (len(hits[i])+len(miss[i])+len(false[i])+len(nan[i])),
                                  len(nan[i]),dps_lut[0][case_std] - len(nan[i])]
                    
                    hit_c       += len(hits[i])
                    miss_c      += len(miss[i])
                    false_c     += len(false[i])
                    cor_neg     += dps_lut[0][case_std] - (len(hits[i])+len(miss[i])+len(false[i])+len(nan[i]))
                    nan_c       += len(nan[i])
                    ops         += dps_lut[0][case_std]
                
            # Store total results
            total = [hit_c, miss_c, false_c, cor_neg, 
                      hit_c + false_c, miss_c+ cor_neg,
                      hit_c + miss_c, false_c + cor_neg,
                      ops, nan_c, ops-nan_c]
            
            # Save results
            total_grid = pd.DataFrame(results).T
            total_grid.columns =['Hit', 'Miss', 'False_pos','Correct_neg', 'no. NaN', 'Total-nan']
    
            
            total_grid.to_csv(f'{dat_out}{case_std}_grid_dich.csv')
            np.save(f'{dat_out}{case_std}_dich_dps', total) 
        
        if ecdf == True:
            sort_list[case_std]    = np.sort(obs_grp[obs_grp.us >0].groupby(['uni_id1','sim_date'])['us'].max())
            
            np.save(f'{dat_out}obs_cdf_1', sort_list)
        #---------------------------------- end of chunk group  ------------------------------------------------------------------------------                                                    
            

        obs_grp.to_csv(f'{dat_out}{case_std}_processed_obs_n.csv')

              
#===================================================================================================================================
#----------------------------------------Analysis-----------------------------------------------------------------------------------
#===================================================================================================================================    
#===================================================================================================================================
#-------------------------------------- Contingency Table -----------------------------------------------------------
#===================================================================================================================================

if cont_table == True:
#====================================== Set plot parameters ====================================================================

    # loop through total list and extract each regions results for plotting   
    results = []
    index = []
    for i in case_list:      
        reg         = np.load(f'{dat_out}{i}_dich_dps.npy', allow_pickle = 'TRUE')#.item()
 
        results.append([x for x in reg])            
        index.append(i)
        
        # Create results dataframe
        results_df          = pd.DataFrame(results)
        results_df.index    = index
#-------------------------------------- Save results ---------------------------------------------------------------------------                
    
    def perc (x, y):
        return(round((x/y)*100,2))
    
    # produce dataframe with raw values
    cont_num = pd.DataFrame({'Fore Yes': [sum(results_df[0]),sum(results_df[2]),sum(results_df[4])],
                             'Fore No': [sum(results_df[1]),sum(results_df[3]),sum(results_df[5])],
                             'Total': [sum(results_df[6]),sum(results_df[7]),sum(results_df[10])]}, 
                             index = ['Observation Yes', 'Observation No', 'Total'])
    
    cont_num.to_excel(f'{dat_out}contingency_num.xlsx')   
    
    # Produce dataframe with percentage values
    cont_tab = pd.DataFrame({'Fore Yes': [perc(sum(results_df[0]),sum(results_df[10])),perc(sum(results_df[2]),sum(results_df[10])),perc(sum(results_df[4]), sum(results_df[10]))],
                             'Fore No': [perc(sum(results_df[1]),sum(results_df[10])),perc(sum(results_df[3]),sum(results_df[10])),perc(sum(results_df[5]),sum(results_df[10]))],
                             'Total': [perc(sum(results_df[6]),sum(results_df[10])),perc(sum(results_df[7]),sum(results_df[10])),perc(sum(results_df[10]),sum(results_df[10]))]}, 
                             index = ['Observation Yes', 'Observation No', 'Total'])                    
    cont_tab.to_excel(f'{dat_out}/contingency_perc.xlsx')  
    
    results_df.columns =['Hit', 'Miss', 'False_pos','Correct_neg', 'Mod_yes', 'Mod_no', 'Obs_yes', 'Obs_no', 'Total_ops', 'no. NaN', 'Total-nan']
    results_df.to_csv(f'{dat_out}dichotomous_results.csv')        
#------------------------------------------------------------------------------------------------------------------------------- 
#===================================================================================================================================
#-------------------------------------- ECDF -----------------------------------------------------------
#===================================================================================================================================

if ecdf == True:
    # Load ecdf data
    ecdf_list    = np.load(f'{dat_out}tot_cdf_1.npy', allow_pickle = 'TRUE').item()
    sort_list    = np.load(f'{dat_out}obs_cdf_1.npy', allow_pickle = 'TRUE').item()
    
    # Specify probability threshold to calibrate sediment entrainment threshold (u*ts)
    thresh = 0.98
    
    # Set plotting parameters
    line = 1.8      # Line width
    sub_font = 18   # Sub-layer axis label font
    top_font = 25   # Main axis label font
    
    # Set sub-plot parameters
    fig = plt.figure(figsize=(13,14))
    gs = gridspec.GridSpec(2,1, wspace=0.15, hspace=0.1)
    
    fig_1 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs[0], wspace=0.1, hspace=0.2)
    fig_2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs[1], wspace=0.1, hspace=0.2)
    
    ax1 = fig.add_subplot(fig_1[0])
    ax0 = fig.add_subplot(fig_2[0])
    
    # Set empty containers to store results for plotting
    thresh_dict     = {}
    perc_dict       = {}
    cdf_mean        = {}
    sort_mean       = {}
    tot_list        = []
    obs_list        = []
#------------------------------------------------------------------------------------------------------------------------------- 
                                                ####### All days data ##########    
    # Loop through results for each region
    for i in case_list:
        # Sort us* results for as x axis data
        x = np.sort(ecdf_list[i])
        # Create y axis list values between 0 and 1 with the numbe rof increments the same length as the x-axis
        y = np.arange(1, len(x)+1) / len(x)
        # Identify position within the y array at which the ecdf exceeds the variable threshold
        position = np.argmax(y > thresh)
        # Plot ecdf
        ax1.plot(x, y,  linestyle = '-', label = f'{cas_reg_di[i]}', lw=line)
        # add us* value at which the ecdf crosses the variable probability threshold
        thresh_dict[i]  = x[position]
        # add proportion of ecdf exceeding 0.2 threshold
        perc_dict[i]    = [round(y[np.argmax(x > 0.2)],2)]
        # add the calcualted mean us* value of ecdf
        cdf_mean[i]     = mean(ecdf_list[i])
        # Add every thenth value from sorted list of values to a global container
        tot_list.extend(x[::10])
    
    # Sort globally collated list of us* values
    x = np.sort(tot_list)  
    # Create y axis list values between 0 and 1 with the numbe rof increments the same length as the x-axis
    y = np.arange(1, len(x)+1) / len(x)
    # Plot ecdf for globally collated ecdf
    ax1.plot(x, y,  linestyle = '--',color = 'k', label = 'Combined cdf', lw=line)
    # Identify position within the y array at which the ecdf exceeds the variable threshold
    position = np.argmax(y > thresh)
    
    # Draw v line describing the position of the globally fixed threshold 0.2
    ax1.vlines(0.2,0,1, lw=line)
    # Draw horizontal line for the position of the variable probability threshold
    ax1.hlines(thresh,0,0.6, ls = '-', color = 'blue', lw=line)
    # Plot verticale dashed line through both subplots describing the required threshold to meet variable probability
    ax1.vlines(x[position],0,thresh, lw=line, ls = '--', color='red', label = '98% Threshold')
    ax0.vlines(x[position],0,1, lw=line, ls = '--', color='red', label = '98% Threshold')
    
    # add us* value at which the glaobl ecdf crosses the variable probability threshold
    thresh_dict['Total']  = x[position] 
    # add proportion of global ecdf exceeding 0.2 threshold
    perc_dict['Total']    = [round(y[np.argmax(x > 0.2)],2)]
    
    # Add legend to upper subplot
    ax1.legend(loc = 'lower right', fontsize = 14)
    # Set y axis label
    ax1.set_ylabel('ecdf (All results)',fontsize=top_font) 
    # Set x axis limits
    ax1.set_xlim(0,0.6)
    # Set plot area background colour
    ax1.set_facecolor('#F9F8F4')
    # Set axis labele paramaters
    ax1.tick_params(labelcolor="k", labelsize= sub_font)
    # Add text label
    textstr1 = 'A.'
    # Position text in subplot
    ax1.text(-0.05, 1.02, textstr1, transform=ax1.transAxes, fontname="Arial",fontweight='bold', fontsize=30,
                verticalalignment='top', horizontalalignment = 'right')
#-------------------------------------------------------------------------------------------------------------------------------     
                                             ####### Observed days data ##########  
    # Loop through results for each region
    for i in case_list:
        # Sort us* results for as x axis data
        x = sort_list[i]
        # Create y axis list values between 0 and 1 with the numbe rof increments the same length as the x-axis
        y = np.arange(1, len(sort_list[i])+1) / len(sort_list[i])
        # Identify position within the y array at which the ecdf exceeds the variable threshold
        position = np.argmax(x > thresh_dict[i])
        # Plot ecdf
        ax0.plot(x, y,  linestyle = '-', label = f'{cas_reg_di[i]}', lw=line)
        # add the calcualted mean us* value of ecdf
        sort_mean[i]    = mean(sort_list[i])
        # add proportion of ecdf exceeding 0.2 threshold and variable threshold
        perc_dict[i].extend([round(y[np.argmax(x > 0.2)],2), round(y[np.argmax(x > thresh_dict['Total'])],2)])
        # Add every each value from sorted list of observed day values to a global container
        obs_list.extend(x)
        
    
    # Sort globally collated list of us* values
    x = np.sort(obs_list)   
    # Create y axis list values between 0 and 1 with the numbe rof increments the same length as the x-axis
    y = np.arange(1, len(x)+1) / len(x)
    # Plot ecdf for globally collated ecdf
    ax0.plot(x, y,  linestyle = '--',color = 'k', label = 'Observed total', lw=line)
    # Identify position within the y array at which the ecdf exceeds the variable threshold
    position = np.argmax(y > thresh)
    
    # Draw v line describing the position of the globally fixed threshold 0.2
    ax0.vlines(0.2,0,1, lw=line)      
    
    # add zero values percentage list to main list length    
    perc_dict['Total'].extend([0,0])
    
    # Set y axis label    
    ax0.set_ylabel('ecdf (Observed Yes)',fontsize=top_font)
    # Set x axis limits
    ax0.set_xlim(0,0.6)
    # Set plot area background colour
    ax0.set_facecolor('#F9F8F4')
    # Set x axis label    
    ax0.set_xlabel('u$_{s*}$',fontsize=top_font)
    # Set axis labele paramaters    
    ax0.tick_params(labelcolor="k", labelsize= sub_font)
    # Add text label
    textstr2 = 'B.'
    # Position text in subplot
    ax0.text(-0.05, 1.02, textstr2, transform=ax0.transAxes, fontname="Arial",fontweight='bold', fontsize=30,
                verticalalignment='top', horizontalalignment = 'right')
    
    # Data sheet
    results = pd.DataFrame(perc_dict).T
    results['thresh'] = results.index.map(thresh_dict)
    results['s_mean'] = results.index.map(sort_mean)
    results['e_mean'] = results.index.map(cdf_mean)
    
    results.columns =['% False pos.','% Hit',f'{thresh}% % Hit (var)',f'{thresh}% us* thresh', 'obs_mean', 'total_mean']
    for c in range(0,3):
        results[f'{results.columns[c]}'] = results.apply(lambda x: 100-(x[c] * 100), axis = 1)
        
    results         = results[['% False pos.','% Hit',f'{thresh}% us* thresh',f'{thresh}% % Hit (var)', 'obs_mean', 'total_mean']]
    
    results.to_excel(f'{dat_out}thresh_{thresh}_res.xlsx')
    
    
    