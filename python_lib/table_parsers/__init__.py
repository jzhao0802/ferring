# -*- coding: utf-8 -*-

from table_parsers.medical_history_parser import MedicalHistoryParser
from table_parsers.delivery_data_parser import DeliveryDataParser
from table_parsers.exposure_parser import ExposureParser

table_parser_map = {
        'mh': MedicalHistoryParser,
        'dd': DeliveryDataParser,
        'ex': ExposureParser
        }