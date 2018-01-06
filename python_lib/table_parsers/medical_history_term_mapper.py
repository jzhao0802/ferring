import pandas as pd


class MedicalHistoryTermMapper(object):

    def __init__(self):
        #Brief explanation of regexes:
        # (?=*) GROUPS REPRESENT AND GROUPS
        # (?=^((?!*).)*$) represents negated string group
        # (?!^*$) NEGATES EXACT EXPRESSION
        self._condition_map = {
            'GBS POSITIVE': '(?=.*(GROUP|GBS))(?=^((?!UNKNOWN).)*$)',
            'SMOKER': 'TOBACCO|SMOKE',
            'BIPOLAR DISORDER': 'BIPOLAR',
            'DEPRESSION': '(?=.*DEPR)(?=^((?!BIPOLAR).)*$)',
            'OLIGOHYDRAMNIOS': 'OLIGOHYDRAMNIOS',
            'ASTHMA': 'ASTHMA',
            'OBESITY': 'OBES',
            'GESTATIONAL HYPERTENSION': '(?=.*HYPERTENSION)(?=PREGNANCY|GESTATIONAL)',
            'HYPERTENSION': '(?=.*HYPERTENSION)(?=^((?!PREGNANCY|GESTATIONAL).)*$)',
            'GESTATIONAL DIABETES': '(?=.*DIABETES)(?=^((?!(CLASS C)|(CLASS B)|(TYPE I)|(TYPE II)|(TYPE 1)|(TYPE 2)|(LAST PREGNANCY)).)*$)(?!^DIABETES$)(?!^DIABETES B$)(?!^DIABETES (MELLITUS|MELLITIS)$)(?!^INSULIN DEPENDENT DIABETES$)',
            'GESTATIONAL DIABETES PREV PREG': '(?=.*DIABETES)(?=.*LAST)(?=^((?!(CLASS C)|(CLASS B)|(TYPE I)|(TYPE II)|(TYPE 1)|(TYPE 2)).)*$)(?!^DIABETES$)(?!^DIABETES B$)(?!^DIABETES (MELLITUS|MELLITIS)$)(?!^INSULIN DEPENDENT DIABETES$)',
            'DIABETES': '((?=.*DIABETES)(?=.*((CLASS C)|(CLASS B)|(TYPE I)|(TYPE II)|(TYPE 1)|(TYPE 2))))|(^(DIABETES|DIABETES B|DIABETES (MELLITUS|MELLITIS)|INSULIN DEPENDENT DIABETES)$)',
            'PRE_ECLAMPSIA': '(?=.*ECLAMPSIA)(?=^((?!PREVIOUS PREGNANCY).)*$)',
            'PRE_ECLAMPSIA PRIOR PREGNANCY': '(?=.*ECLAMPSIA)(?=.*PREVIOUS PREGNANCY)',
            'CHLAMYDIA': 'CHLAMYDIA',
            'POST_TERM PREGNANCY': 'POST-TERM PREGNANCY \(>=40W\)',
            'URINARY TRACT INFECTION': 'URINARY|UTI',
            'YEAST INFECTION': 'YEAST',
            'LARGE BABY': '(?=.*LARGE)(?=(^((?!THYROID).)*$))',
            'ANEMIA': 'ANEMIA',
            'ANXIETY': 'ANXIETY',
            'BACTERIAL VAGINOSIS': 'BACTERIAL',
            'DECREASED FETAL MOVEMENT': 'DECREASED FETAL MOVEMENT',
            'FETAL MACROSOMIA': 'MACROSOMIA',
            'HERPES': 'HERP',
            'SMALL FOR GESTATIONAL AGE': 'SMALL FOR'
            #'ANEMIA': '(?=.*ANEMIA)(?=(^((?!PREGNANCY).)*$))',
            #'ANEMIA PREGNANCY': '(?=.*ANEMIA)(?=.*PREG)'
        }

        self._matched_terms = {}

    def get_condition_map(self):
        return self._condition_map

    def log_mapped_terms(self, log_path):
        spreadsheet = pd.ExcelWriter(log_path)
        for term,df in self._matched_terms.items():
            #df = pd.DataFrame(matched_terms)
            df.to_excel(spreadsheet, term)
            #df.to_excel(spreadsheet, term, index=False)
        spreadsheet.save()
        spreadsheet.close()

    def get_matched_terms(self):
        return self._matched_terms

    def map_terms(self, df, term_column, primary_key, num_patients):
        df.loc[:, term_column] = df[term_column].str.strip().str.upper()
        for term,term_pattern in self._condition_map.items():
            matches = df[term_column].str.contains(term_pattern)
            grouped_terms = df[matches].drop_duplicates(subset=[term_column, primary_key]).groupby(term_column).size().sort_values(ascending=False)
            grouped_terms = grouped_terms.reset_index().rename(columns={0: 'Count', term_column: 'Term'})
            grouped_terms['Prevelance'] = 100*(grouped_terms['Count']/num_patients)
            grouped_terms['Percentage of Term'] = 100 * (grouped_terms['Count'] / grouped_terms['Count'].sum())
            grouped_terms.loc['Total'] = grouped_terms.sum()
            grouped_terms.loc['Total', 'Term'] = ''
            self._matched_terms[term] = grouped_terms
            df.loc[matches, term_column] = term

    def filter_terms(self, df, term_column):
        return df.loc[df[term_column].str.contains('|'.join(['^%s$'%term for term in self._condition_map.keys()])), :]
