

class MedicalConditionMap(object):

    def __init__(self):
        self._condition_map = {
            'GBS POSITIVE' : 'GBS|GROUP',
            'SMOKER': 'TOBACCO|SMOKE',
            'DEPRESSION': 'BIPOLAR|DEPR',
            'OLIGOHYDRAMNIOS': 'OLIGOHYDRAMNIOS',
            'ASTHMA': 'ASTHMA',
            'OBESITY': 'OBESITY|OBESE',
            'HYPERTENSION': 'HYPERTENSION', #THIS  CAPTURES BOTH PREGANCY INDUCED, CHRONIC, ONE CASE OF PREECLAMPSIA
            'GESTATIONAL DIABETES': ['DIABETES', {'PRE-GESTATIONAL DIABETES': '(CLASS C)|(CLASS B)|(TYPE I)|(TYPE II)'}],
            'PRE-ECLAMPSIA': 'ECLAMPSIA',
            'CHLAMYDIA': 'CHLAMYDIA',
            'POST-TERM PREGNANCY (>=40W)': 'POST-TERM PREGNANCY (>=40W)',
            'URINARY TRACT INFECTION': 'URINARY|UTI',
            'YEAST INFECTION (CANDIDIASIS)': 'YEAST'
        }


    def get_condition_map(self):
        return self._condition_map

