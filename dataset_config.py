helpdesk = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    'Resource',
                                    "seriousness",
                                    "service_level"
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': ["customer",
                                   "product",
                                   "responsible_section",
                                   "service_type",
                                   "support_section"],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': []
}

bpic2015_1 = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "question",
                                    "monitoringResource",
                                    "Resource"],
    'CATEGORICAL_STATIC_COLUMNS': ["Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw', 'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)'
                                   ],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["SUMleges"]
}

bpic2015_2 = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "question",
                                    "monitoringResource",
                                    "Resource"],
    'CATEGORICAL_STATIC_COLUMNS': ["Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw',
                                   'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)'
                                   ],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["SUMleges"]
}

bpic2015_3 = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_COLUMNS': ['Activity'],
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "monitoringResource",
                                    "question",
                                    "Resource",
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': ["Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw',
                                   'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)',
                                   'Flora en Fauna'
                                   ],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["SUMleges"
                                 ]
}

bpic2015_4 = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "monitoringResource",
                                    "question",
                                    "Resource",
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': ["Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',  ## CATEGORICAL
                                   'Bouw',  ## CATEGORICAL
                                   'Brandveilig gebruik (vergunning)',  ## CATEGORICAL
                                   'Gebiedsbescherming',  ## CATEGORICAL
                                   'Handelen in strijd met regels RO',  ## CATEGORICAL
                                   'Inrit/Uitweg',  ## CATEGORICAL
                                   'Kap',  ## CATEGORICAL
                                   'Milieu (neutraal wijziging)',  ## CATEGORICAL
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',  ## CATEGORICAL
                                   'Milieu (vergunning)',  ## CATEGORICAL
                                   'Monument',  ## CATEGORICAL
                                   'Reclame',  ## CATEGORICAL
                                   'Sloop'  ## CATEGORICAL
                                   ],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["SUMleges"]
}

bpic2015_5 = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "monitoringResource",
                                    "question",
                                    "Resource",
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': ["Responsible_actor",
                                   'Aanleg (Uitvoeren werk of werkzaamheid)',
                                   'Bouw', 'Brandveilig gebruik (vergunning)',
                                   'Gebiedsbescherming',
                                   'Handelen in strijd met regels RO',
                                   'Inrit/Uitweg',
                                   'Kap',
                                   'Milieu (neutraal wijziging)',
                                   'Milieu (omgevingsvergunning beperkte milieutoets)',
                                   'Milieu (vergunning)',
                                   'Monument',
                                   'Reclame',
                                   'Sloop',
                                   'Brandveilig gebruik (melding)',
                                   'Milieu (melding)',
                                   'Flora en Fauna'
                                   ],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["SUMleges"]
}

sepsis = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    'org:group',
                                    'DiagnosticArtAstrup',
                                    'DiagnosticBlood',
                                    'DiagnosticECG',
                                    'DiagnosticIC',
                                    'DiagnosticLacticAcid',
                                    'DiagnosticLiquor',
                                    'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                                    'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                                    'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                                    'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                                    'SIRSCritTemperature', 'SIRSCriteria2OrMore', 'Diagnose'
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': [],
    'NUMERICAL_DYNAMIC_COLUMNS': ['CRP',
                                  'LacticAcid',
                                  'Leucocytes',
                                  'Age'],
    'NUMERICAL_STATIC_COLUMNS': []
}

bpic2012a = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    'Resource'],
    'CATEGORICAL_STATIC_COLUMNS': [],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ['AMOUNT_REQ']
}

bpic2012o = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    'Resource'],
    'CATEGORICAL_STATIC_COLUMNS': [],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ['AMOUNT_REQ']
}

bpic2012w = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    'Resource'],
    'CATEGORICAL_STATIC_COLUMNS': [],
    'NUMERICAL_DYNAMIC_COLUMNS': ['proctime'],
    'NUMERICAL_STATIC_COLUMNS': ['AMOUNT_REQ']
}

credit = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_COLUMNS': ['Activity'],
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity'],
    'CATEGORICAL_STATIC_COLUMNS': [],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': []
}

hospital = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "Resource",
                                    "actOrange",
                                    "actRed",
                                    # "blocked",--> Do not use since only value of 0. occurs --> Not informative
                                    "caseType",
                                    "diagnosis",
                                    "flagC",
                                    "flagD",
                                    # "msgCode",--> Do not use since only value of 0. occurs --> Not informative
                                    # "msgType",--> Do not use since only value of 0. occurs --> Not informative
                                    "state",
                                    "version"
                                    ],
    'CATEGORICAL_STATIC_COLUMNS': ["speciality"],
    'NUMERICAL_DYNAMIC_COLUMNS': [# "msgCount" --> Do not use since only value of 0. occurs --> Not informative
                                  ],
    'NUMERICAL_STATIC_COLUMNS': []
}

invoice = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ['Activity',
                                    "Resource",
                                    "ActivityFinalAction",
                                    "EventType"],
    'CATEGORICAL_STATIC_COLUMNS': ["CostCenter.Code",
                                   "Supplier.City",
                                   "Supplier.Name",
                                   "Supplier.State"],
    'NUMERICAL_DYNAMIC_COLUMNS': [],
    'NUMERICAL_STATIC_COLUMNS': ["InvoiceTotalAmountWithoutVAT"]
}

Production_Data = {
    'CASE_ID_COLUMN': 'Case ID',
    'ACTIVITY_COLUMN': 'Activity',
    'TIMESTAMP_COLUMN': 'Complete Timestamp',
    'CATEGORICAL_DYNAMIC_COLUMNS': ["Activity",
                                    "Resource",
                                    "Report Type",
                                    "Worker ID"],
    'CATEGORICAL_STATIC_COLUMNS': ["Part Desc"],
    'NUMERICAL_DYNAMIC_COLUMNS': ["Qty Completed",
                                  "Qty for MRB"],
    'NUMERICAL_STATIC_COLUMNS': ["Work Order Qty"]
}
