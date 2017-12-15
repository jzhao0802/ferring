
# coding: utf-8

# In[139]:


import pandas as pd
import set_lib_paths
import numpy as np
import seaborn as sns
import copy
sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#Define pandas defaults
pd.options.display.max_rows = 15
pd.options.display.max_columns = 40


#Define function to plot distributions from grouped data
def plot_multi(df, grouped_df, cols, cust_labels, group_keys, group_col, suffix='', kde=False):
        #Get current sns colour palette
        current_palette = sns.color_palette()
        if kde:
            suffix += '_kde'
        n_patients = len(df)
        plt.clf()
        #sns.distplot(df['GESTATIONAL_AGE_DAYS'][df['STUDY_CODE'] == 303].dropna(), kde=False)
        bs_cols = list(df.filter(regex='^BS_').keys())
        #
        #for col in bs_cols:
        #    sns.distplot(df[col].dropna(), label=col.split('_')[1], kde=False, bins=range(12), hist_kws={"alpha":0.4})
        for col,x_title in plot_cols.items():
            plt.clf()
            if col != 'RACE':        f, (ax_box1, ax_box2, ax_box3, ax_hist) = plt.subplots(4, sharex=True,
                                            gridspec_kw={"height_ratios": (.05, .05, .05, .85)})
            #else:
            #    f, (ax_hist) = plt.subplots(1, sharex=True, gridspec_kw={"height_ratios": (1)})

            if col in ['BS_BASELINE', 'RACE']:
                if kde: continue
                #ax = sns.distplot(df[col].dropna(), kde=False, bins=range(12), hist_kws={"alpha":0.4})
                #[df[col].dropna()]
                if col == 'BS_BASELINE':
                    #df.groupby(['STUDY_CODE', col]).size().apply(lambda x:100*(x/n_patients)).unstack(0).plot.bar(stacked=True, edgecolor='white', ax=ax_hist)
                    df.groupby([group_col, col]).size().unstack(0).plot.bar(stacked=True, edgecolor='white', ax=ax_hist)
                else:
                    df.loc[df['RACE']=='BLACK_OR_AFRICAN_AMERICAN', 'RACE'] = 'BLACK/AA'
                    ax = df.groupby([group_col, col]).size().unstack(0).plot.bar(stacked=True, edgecolor='white', rot=0)
            else: #ax = sns.distplot(df[col].dropna(), kde=False, hist_kws={"alpha":0.4})
                #ax = df.groupby(['STUDY_CODE', col]).size().unstack(0).plot.hist(stacked=True, edgecolor='white')
                #df[col].plot.hist(edgecolor='white')
                n_bins = 10
                if 'TIME_DELTA' in col: n_bins = 30
                if not kde:
                    df_c = pd.DataFrame({cust_labels[0]: grouped_df.get_group(group_keys[0])[col], cust_labels[1] :  grouped_df.get_group(group_keys[1])[col]})
                    plot = df_c.plot.hist(stacked=True, edgecolor='white', ax=ax_hist, bins=n_bins)
                else:
                    print(col)
                    sns.kdeplot(grouped_df.get_group(group_keys[0])[col], label=cust_labels[0], shade=True)
                    sns.kdeplot(grouped_df.get_group(group_keys[1])[col], label=cust_labels[1], shade=True)
                    plot = sns.kdeplot(df[col], label='COMBINED', shade=True)
            #ax.text(0.3, 1.3,'Patients with missing information: %.2f%%'%((n_missing*100)/len(df)))

            if col == 'RACE':
                plt.xlabel(x_title)
                plt.ylabel('Number of Patients')

                handles, labels = ax.get_legend_handles_labels()
                labels[0]  = cust_labels[0]
                labels[1]  = cust_labels[1]
                #labels[0] = 'MISO-OBS-004'
                #labels[1] = 'MISO-OBS-303'
                print(labels)
                plt.legend(handles, labels)
                plt.savefig('F:/Projects/Ferring/results/pre_modelling/sprint_1_2_plots/' + col.lower() + suffix + '.svg', format='svg', dpi=1200)
                continue
            #ax2 = ax.twinx()
            sns.boxplot(x=df[df['STUDY_CODE']==4][col].dropna(), ax=ax_box2, showmeans=True, color=current_palette[0])
            sns.boxplot(x=df[df['STUDY_CODE']==303][col].dropna(), ax=ax_box1, showmeans=True, color=current_palette[1])
            bx = sns.boxplot(x=df[col].dropna(), ax=ax_box3, showmeans=True, color=current_palette[3])

            ax_box3.set(yticks=[])
            ax_box1.set(yticks=[])
            ax_box2.set(yticks=[])


            handles, labels = ax_hist.get_legend_handles_labels()
            plt.ylabel('Number of Patients')
            if handles and not kde:
                handles.append(copy.copy(handles[-1]))
                try: handles[-1].set_facecolor(current_palette[3])
                except:
                    handles[-1] = copy.copy(prev_handles[-1])
                    handles[-1].set_facecolor(current_palette[3])
                if 'BS' in col:
                    labels[0]  = cust_labels[0]
                    labels[1]  = cust_labels[1]
                    #plt.ylabel('Number of Patients')
                    #for item in handles[-1].get_children():
                    #    item.set_facecolor(current_palette[3])

                labels.append('COMBINED')
                plt.legend(handles, labels)

            prev_handles = handles
            #ax2.set(ylim=(-10,5))
            #n_missing = df[col].isnull().sum()

            plt.xlabel(x_title)

            plt.savefig('F:/Projects/Ferring/results/pre_modelling/sprint_1_2_plots/' + col.lower() + suffix + '.svg', format='svg', dpi=1200)
            #sns.violinplot(x='')
            #plt.legend()


#Load merged data file
case_study_codes = ['303', '004']
merged_file = 'F:/Projects/Ferring/data/pre_modelling/merged_data/MERGED_COUNT_DATE.csv'
df = pd.read_csv(merged_file)
#Remove patients that did not complete study...
df = df[df['COMPLETED'] & ~df['COMPLETED'].isnull()]
#Add gestational age in weeks
df['GESTATIONAL_AGE_WEEKS'] = df['GESTATIONAL_AGE_DAYS']/7

#Combine vaginal delivery methods
df.loc[df['DD_DELIVERY_METHOD'].str.contains('VAGINAL'), 'DD_DELIVERY_METHOD'] = 'VAGINAL'

print('mean age is :', df['AGE'].mean())


#Create time to oxytocin admin col
df['TIME_DELTA_EX_START_OXYTOCIN_ADMIN'] = (pd.to_datetime(df['OAENTPT']) - pd.to_datetime(df['EX_START_TIME']))/ np.timedelta64(1, 'h')

#Create time to onset of labour column
df['TIME_DELTA_EX_START_ONSET_LABOUR'] = (pd.to_datetime(df['ACTIVE LABOR']) - pd.to_datetime(df['EX_START_TIME']))/ np.timedelta64(1, 'h')
print(df['TIME_DELTA_EX_START_ONSET_LABOUR'].mean())

#87 patients missing onset of labour info - did not reach active labour??
print(df['TIME_DELTA_EX_START_ONSET_LABOUR'].isnull().sum())


#Set column for label based on definition agreed with Ferring
df['label'] = (df['TIME_DELTA_EX_START_ONSET_LABOUR'] <=24) & (df['DD_DELIVERY_METHOD'] == 'VAGINAL')
df['label'] = df['label'].astype(int)


# # Produce plots
#Standard distribution plots
plot_cols = {'AGE':'Patient Age', 'BS_BASELINE': 'Baseline Modified Bishop Score', 'OXYTOCIN_DOSAGE': 'Pre-delivery Oxytocin Dosage [Units]', 'BMI':'BMI', 'HEIGHT':'Height [m]', 'WEIGHT': 'Weight [kg]', 'GESTATIONAL_AGE_DAYS': 'Gestational Age [days]', 'RACE': 'Ethnicity', 'GESTATIONAL_AGE_WEEKS': 'Gestational Age [weeks]', 'TIME_DELTA_EX_START_OXYTOCIN_ADMIN': 'Time between Propess administration and Oxytocin administration [hours]', 'TIME_DELTA_EX_START_ONSET_LABOUR': 'Time between Propess administration and onset of Active Labour [hours]'}

plot_multi(df, grouped_df, plot_cols, ['MISO-OBS-004', 'MISO-OBS-303'], [4, 303], 'STUDY_CODE')

#Produce plots grouped by label
grouped_df_dm = df.groupby('label')

plot_multi(df, grouped_df_dm, plot_cols, ['SUCCESS', 'FAILURE'], [1, 0], 'label', suffix='_outcome')
plot_multi(df, grouped_df_dm, plot_cols, ['SUCCESS', 'FAILURE'], [1, 0], 'label', suffix='_outcome', kde=True)

#Produce tables of outcome x oxytocin
print(df.groupby(['label', 'OXYTOCIN_ADMINISTERED']).size().unstack(0).transpose()*100/n_patients)
print(df.groupby(['label', 'OXYTOCIN_ADMINISTERED']).size().unstack(0).transpose())

#Get number of patients with BMI > 50
(df['BMI'] > 50).sum()
(df[df['STUDY_CODE'] == 303]['BMI'] > 50).sum()
(df[df['STUDY_CODE'] == 4]['BMI'] > 50).sum()

#Get mean gestational age
print(df['GESTATIONAL_AGE_WEEKS'].mean())

#Print mean time delta from administration to oxytocin admin time
df['TIME_DELTA_EX_START_OXYTOCIN_ADMIN'].mean()

