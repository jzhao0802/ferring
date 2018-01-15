


class ModellingDatasetCleaner(object):
    '''
        Class to remove variables not necessary for modelling, and to create reference class for all dummy variables
    '''

    def __init__(self):
        super().__init__()
        self._features_to_remove = [
            '^BS_12$', '^BS_18$', '^BS_24$', '^BS_6$', '^COMPLETED$', '^DD_DELIVERY_TIME$', '^DISCHARGE_TIME$', '^EX_END_TIME$',
            '^TIME_DELTA_EX_START_OXYTOCIN_ADMIN$', '^TIME_DELTA_EX_START_ONSET_LABOUR$', '^STUDY_CODE$', '^SUBJID$',
            '^MODE_OF_DELIVERY$', '^OXYTOCIN_ADMINISTERED$', '^OXYTOCIN_ADMINISTRATION_TIME$', '^OXYTOCIN_DOSAGE$',
            '^RACE$', '^ACTIVE_LABOUR_TIME$', '^COUNTRY$', '^RACE_RAW$', '^RACE_PROCESSED$', '^EX_START_TIME$', '^APGAR.*', '^X1.*',
            '^Unnamed.*', '^USUBJID$'
        ]

        self._dummy_var_prefixes = [
            'RACE',
            'COUNTRY',
            'PREGTYPE'

        ]

    def remove_nan_BS(self, df):
        return df[~df['BS_BASELINE'].isnull()]

    def remove_ref_dummy_class(self, df, dummy_prefix):
        '''
        Method to create reference class for a dummy variable
        :param df: dataframe representing flatfile
        :param dummy_prefix: prefix of dummy variable to create reference class for
        :return: dummy variable used as reference class
        '''
        dummy_vars = list(df.filter(regex='^%s'%dummy_prefix).keys())
        del df[dummy_vars[0]]
        return dummy_vars[0]

    def remove_all_ref_dummy_classes(self, df):
        '''
        Create reference classes for all dummy variables
        :param df: dataframe representing flatfile
        :return: list of dummy variables used as reference classes
        '''
        return [self.remove_ref_dummy_class(df, dummy) for dummy in self._dummy_var_prefixes]

    def convert_gestational_age_to_weeks(self, df):
        df.loc[:,'GESTATIONAL_AGE_WEEKS'] = round(df['GESTATIONAL_AGE_DAYS']/7)
        del df['GESTATIONAL_AGE_DAYS']


    def select_modelling_features(self, df):
        '''
        Remove variables no relevant to modelling
        :param df: dataframe representing flatfile
        :return: dataframe with variables not relevant to modelling remove
        '''
        return df.filter(regex='(?=(^((?!%s).)*$))'%('|'.join(self._features_to_remove)))

    def merge_race_dummy_classes(self, df, merge_list=[]):
        #print (df.filter(regex='^RACE'))
        merged = 'RACE_dummy_' + '_'.join(merge_list)
        #df.insert(-1, merged, False)
        df[merged] = False
        for dummy in merge_list:
            df[merged] = df['RACE_dummy_%s'%dummy] + df[merged]
            del df['RACE_dummy_%s'%dummy]
        #print (df[merged])


    def remove_sparse_variables(self, df, threshold, label_col, thresh_type='both'):
        if thresh_type == 'pos': df_thresh = df[df[label_col] == 1]
        elif thresh_type == 'neg': df_thresh = df[df[label_col] == 0]
        else: df_thresh = df

        tot_vals = ((~df_thresh.isnull()) & (df_thresh>0)).sum()
        tot_vals = tot_vals.sort_values(ascending=True).reset_index()
        vars_to_remove = tot_vals[tot_vals[0]<=10]['index']
        for var in vars_to_remove: del df[var]
        return vars_to_remove

    def clean_data_for_modelling(self, df, sparse_var_threshold=10, label_col='LABEL', thresh_type='both', gs_age_weeks=False, merge_race_dummies=[]):
        '''
        Remove variables not necessary for modelling and create reference class for dummy variables
        :param df: dataframe representing flatfile
        :return: tuple -
            first element - dataframe with irrelevant variables removed and reference classes for dummy variables removed.
            second element - dummy variable used as reference class.
        '''
        if merge_race_dummies:
            self.merge_race_dummy_classes(df, merge_list=merge_race_dummies)
        df = self.remove_nan_BS(df)
        df = self.select_modelling_features(df)
        ref_dummy_variables = self.remove_all_ref_dummy_classes(df)
        removed_vars = self.remove_sparse_variables(df, sparse_var_threshold, label_col, thresh_type=thresh_type)
        if gs_age_weeks:
            self.convert_gestational_age_to_weeks(df)
        return (df, ref_dummy_variables, removed_vars)

