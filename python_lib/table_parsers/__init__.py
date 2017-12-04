# -*- coding: utf-8 -*-

from table_parsers.medical_history_parser import MedicalHistoryParser
from table_parsers.delivery_data_parser import DeliveryDataParser
from table_parsers.exposure_parser import ExposureParser
from table_parsers.bishop_score_parser import BishopScoreParser
from table_parsers.demographic_parser import DemographicParser
from table_parsers.vital_stats_parser import VitalStatsParser

table_parser_map = {
        'mh': MedicalHistoryParser,
        'dd': DeliveryDataParser,
        'ex': ExposureParser,
        'bs': BishopScoreParser,
        'dm': DemographicParser,
        'vs': VitalStatsParser
        }