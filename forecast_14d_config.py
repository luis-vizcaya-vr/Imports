from datetime import datetime, timedelta
import pandas as pd
import pyodbc


db_credential_dict = {'sids': """server=SIMON;database=EWP;uid=epreader;pwd=Envapower09""",
                      'wind': """server=WindProdDB;""",
                      'seer': """server=seerprod;database=SEER_{SEER_ISO};uid=epreader;pwd=()&TUq853g""",
                      'zephyr': """server=genpowerdemand.internal.genscape.com;database=PowerForecasting""" 
                      }

forecast_input_dict = {'PJM': {'start_date': datetime(2021,12,1),
                                'end_date': datetime(2021,12,1)+timedelta(hours = 23, minutes = 59),
                                'outages': 10000,
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'blue_steel_demand_PJM.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {4354: 'MidA', 6182: 'South', 4355: 'West', 6795: 'COMED'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'imports': {'MISO': 10000, 'NYISO': 500},
                                'hub_price_sids_dict': {4463: 'WHUB', 4460: 'EHUB', 5516: 'ADHUB', 4887: 'NIHUB', 4456:'AE', 5511:'AEP',  4458:'APS', 54614: 'ATSI',  45057: 'BGE', 4450:'PPL', 4448: 'PSEG', 4459:'RECO',  
                                45058: 'COMED', 45059:'DAYTON', 59564: 'DEOK', 6000:'DOM', 4457:'DPL', 5773:'DUQ', 9374:'EKPC', 45063: 'JCPL', 4453: 'METED', 4449: 'PE',4455: 'PEP', 4454:'PN'},
                                
                                'fuel_price_dict': {'BIT - NAPP': 3.5,
                                                    'BIT - CAPP': 3.5,
                                                    'BIT - ILB': 2.5,
                                                    'NG - Chicago City Gate': 5.13,
                                                    'NG - Dominion South': 5.35,
                                                    'NG - TETCO M3 NY': 5.70,
                                                    'NG - Transco-Z6 (non-NY)': 5.70,
                                                    'NG - Transco-Z5 (non-WGL)': 6.0,
                                                    },
                                'zonal_fuel_mapping_file': 'zonal_fuel_mapping.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                              }


backtest_input_dict = {'ISONE': {'start_date': datetime(2020,6,1),
                                'end_date': datetime(2020,6,3)+timedelta(hours = 23, minutes = 59),
                                'outage_sid': [4540],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {3979: 'isone'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {4390: 'isone', 14: 'isone', 22: 'isone', 30: 'isone', 38: 'isone', 9451: 'isone'},
                                'hub_price_sids_dict': {4108: 'masshub'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'NYISO': {'start_date': datetime(2022,1,1),
                                'end_date': datetime(2022,1,1) + timedelta(hours = 23, minutes = 59),
                                'outage_sid': [62896],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {64473: 'NYZA', 64474: 'NYZB', 64475: 'NYZC', 64476: 'NYZD', 64477: 'NYZE', 64478: 'NYZF', 64479: 'NYZG', 64480: 'NYZH', 64481: 'NYZI', 64482: 'NYZJ', 64483: 'NYZK'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {4728: 'NYZE', 4729: 'NYZD', 4730: 'NYZG', 4731: 'NYZF'},  # 4728: 'IMO', 4729: 'HQ', 4730: 'PJM', 4731: 'NPX'
                                'hub_price_sids_dict': {4363: 'NYZA', 4369: 'NYZG', 4372: 'NYZJ', 4373: 'NYZK'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'PJM': {'start_date': datetime(2021,7,1),
                                'end_date': datetime(2021,7,1) + timedelta(hours = 23, minutes = 59),
                                'outage_sid': [77937],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                'demand_file': 'blue_steel_demand_PJM.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {4336: 'MidA', 6023: 'South', 6170: 'West', 64374: 'COMED'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {7732: 'MISO', 6096: 'NYISO'},
                                'hub_price_sids_dict': {4463: 'WHUB', 4460: 'EHUB', 5516: 'ADHUB', 4887: 'NIHUB', 4456:'AE', 5511:'AEP',  4458:'APS', 54614: 'ATSI', 4450:'PPL', 4448: 'PSEG', 4459:'RECO',  
                                4882: 'COMED', 5517:'DAYTON', 59465: 'DEOK', 6000:'DOM', 4457:'DPL', 5773:'DUQ', 65962:'EKPC', 4453: 'METED', 4449: 'PE',4455: 'PEPCO', 4454:'PN'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                #, 45063: 'JCPL',  45057: 'BGE'
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'MISO': {'start_date': datetime(2020,4,2),
                                'end_date': datetime(2020,4,3)+timedelta(hours = 23, minutes = 59),
                                'outage_sid': [78414],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {17211: 'CENTRAL', 17207: 'NORTH', 65767: 'SOUTH'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {9165: 'MISO_NIPS',},
                                'hub_price_sids_dict': {52434: 'INDY', 5971: 'ILL', 5980: 'MICH', 5989: 'MINN', 65913: 'ARK'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'SPP': {'start_date': datetime(2020,4,2),
                                'end_date': datetime(2020,4,3)+timedelta(hours = 23, minutes = 59),
                                'outage_sid': [71096],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {4336: 'MidA', 6023: 'South', 6170: 'West', 64374: 'COMED'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {71096: 'MISO'},
                                'hub_price_sids_dict': {4463: 'whub', 4460: 'ehub', 5516: 'adhub', 4887: 'nihub'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'ERCOT': {'start_date': datetime(2022,5,1),
                                'end_date': datetime(2022,5,1)+timedelta(hours = 23, minutes = 59),
                                'outage_sid': [66082],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                #'demand_sids_dict': {64443: 'North_C', 64447: 'South_C', 64439: 'Coast', 64446: 'Southern', 64442: 'North', 64448: 'West', 64441: 'Far_West', 64440: 'East'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'demand_sids_dict': {64443: 'North_C', 64447: 'South_C', 64439: 'Coast', 64446: 'Southern', 64442: 'North', 64448: 'West'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                #'hub_price_sids_dict': {49415: 'nhub', 49416: 'whub', 49417: 'shub', 49418: 'hhub'},
                                #'hub_price_sids_dict': { 49416: 'whub'},
                                'hub_price_sids_dict': {4463: 'WHUB', 4460: 'EHUB', 5516: 'ADHUB', 4887: 'NIHUB', 4456:'AE', 5511:'AEP',  4458:'APS', 54614: 'ATSI', 4450:'PPL', 4448: 'PSEG', 4459:'RECO',  
                                4882: 'COMED', 5517:'DAYTON', 59465: 'DEOK', 6000:'DOM', 4457:'DPL', 5773:'DUQ', 65962:'EKPC', 4453: 'METED', 4449: 'PE',4455: 'PEPCO', 4454:'PN'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                        'CAISO': {'start_date': datetime(2020,4,2),
                                'end_date': datetime(2020,4,3)+timedelta(hours = 23, minutes = 59),
                                'outage_sid': [78414 ],
                                # 'stack_file': 'pjmStack20192020_dates.csv',
                                # 'demand_file': 'pjm_demand.csv', # This should be more carefully considered. We should be using historically forecast demand and applying zonal weighting to them (in this case we would query for the demand from SIDs)
                                'demand_sids_dict': {64489: 'PGE', 64501: 'SCE', 6170: 'SDGE'}, # These are SIDs for MidA, South, West, COMED demand actuals (in that order)
                                'import_sids_dict': {71406: 'PGE',
                                                    71389: 'PGE',
                                                    71356: 'PGE',
                                                    71386: 'PGE',
                                                    71360: 'PGE',
                                                    71411: 'PGE',
                                                    71372: 'PGE',
                                                    71395: 'PGE',
                                                    71410: 'PGE',
                                                    71416: 'PGE',
                                                    71405: 'PGE',
                                                    71388: 'PGE',
                                                    71415: 'PGE',
                                                    106189: 'PGE',
                                                    106190: 'PGE',
                                                    106187: 'PGE',
                                                    106188: 'PGE',
                                                    105904: 'PGE',
                                                    105881: 'PGE',
                                                    103660: 'PGE',
                                                    78656: 'PGE',
                                                    105882: 'PGE',
                                                    105905: 'PGE',
                                                    103667: 'PGE',
                                                    78657: 'PGE',
                                                    71361: 'SCE',
                                                    71390: 'SCE',
                                                    71385: 'SCE',
                                                    71376: 'SCE',
                                                    71355: 'SCE',
                                                    71391: 'SCE',
                                                    71371: 'SCE',
                                                    71380: 'SCE',
                                                    71402: 'SCE',
                                                    71357: 'SCE',
                                                    71366: 'SCE',
                                                    71367: 'SCE',
                                                    71354: 'SCE',
                                                    103661: 'SCE',
                                                    103662: 'SCE',
                                                    105906: 'SCE',
                                                    105880: 'SCE',
                                                    103665: 'SCE',
                                                    103666: 'SCE',
                                                    103663: 'SCE',
                                                    103664: 'SCE',
                                                    103670: 'SCE',
                                                    103671: 'SCE',
                                                    103659: 'SCE',
                                                    78655: 'SCE',
                                                    },
                                'hub_price_sids_dict': {73001: 'NP', 73002: 'SP'},
                                'fuel_price_file': 'allFuelPrice_1920.csv',
                                # 'wind_curve_file': 'windCurves_PJM.csv',
                                'solar_curve_file': 'solarCurves_PJM.csv'
                                },
                              }


iso_hub_dict = {'ISONE': ['masshub'],
                'NYISO': ['NYZA','NYZG','NYZJ','NYZK'],
                'PJM': ['WHUB','ADHUB','EHUB','NIHUB'],
                'MISO': ['INDY','ILL','MICH','MINN','ARK'],
                'SPP': [],
                'ERCOT': ['nhub','whub','shub','hhub'],
                'CAISO': ['NP','SP']
                }

seer_zone_mapping = {'MISO': {'EAI': 'MISO_AR',
                            'DEI': 'MISO_DEIN',
                            'HE': 'MISO_HE',
                            'AMIL': 'MISO_IL',
                            'CWLP': 'MISO_IL',
                            'SIPC': 'MISO_IL',
                            'IPL': 'MISO_IPL',
                            'ITCT': 'MISO_ITC_DTE',
                            'MDU': 'MISO_MDU',
                            'METC': 'MISO_METC_CONS',
                            'AMMO': 'MISO_MO',
                            'SMEPA': 'MISO_MS',
                            'NIPS': 'MISO_NIPS',
                            'SIGE': 'MISO_SIGE',
                            'CLEC': 'MISO_TXLA',
                            'EES': 'MISO_TXLA',
                            'LAFA': 'MISO_TXLA',
                            'LAGN': 'MISO_TXLA',
                            'LEPA': 'MISO_TXLA',
                            'ALTW': 'MRO_IOWA',
                            'MEC': 'MRO_IOWA',
                            'MPW': 'MRO_IOWA',
                            'DPC': 'MRO_MINN',
                            'GRE': 'MRO_MINN',
                            'MP': 'MRO_MINN',
                            'OTP': 'MRO_MINN',
                            'SMMPA': 'MRO_MINN',
                            'XEL': 'MRO_MINN',
                            'UPPC': 'MRO_WUMS',
                            'ALTE': 'RFC_WUMS',
                            'MGE': 'RFC_WUMS',
                            'WEC': 'RFC_WUMS',
                            'WPS': 'RFC_WUMS',
                            'BREC': 'SERC_BREC'},
                        'ERCOT': {'LZ_AEN': 'West',
                             'LZ_CPS': 'South_C',
                             'LZ_HOUSTON': 'Coast',
                             'LZ_LCRA:': 'South_C',
                             'LZ_NORTH': 'North_C',
                             'LZ_RAYBN': 'North',
                             'LZ_SOUTH': 'Southern',
                             'LZ_WEST': 'West',
                             'MISO': 'North'}
                    }

zonal_import_weightings_dict = {'ISONE': {'HQ': {'HQ': 1}
                                    },
                                'NYISO': {'PJM': {'PJM': 1}
                                    },
                                'PJM': {'MISO': {'MISO': 1},
                                        'NYISO': {'NYISO': 1}
                                    },
                                'MISO': {'MISO_NIPS': {'MISO_NIPS': 1}
                                    },
                                'CAISO': {'PGEB': {'PGE': -0.169364},
                                    'PGSB': {'PGE': -0.142895},
                                    'PGF1': {'PGE': -0.12053},
                                    'PGNP': {'PGE': -0.088599},
                                    'PGZP': {'PGE': -0.074663},
                                    'PGSI': {'PGE': -0.070446},
                                    'PGSF': {'PGE': -0.067711},
                                    'PGP2': {'PGE': -0.059983},
                                    'PGST': {'PGE': -0.044281},
                                    'PGCC': {'PGE': -0.040074},
                                    'PGKN': {'PGE': -0.038247},
                                    'PGFG': {'PGE': -0.03286},
                                    'PGNB': {'PGE': -0.030126},
                                    'PGNC': {'PGE': -0.010781},
                                    'PGHB': {'PGE': -0.009442},
                                    'SCEW': {'SCE': -0.389709},
                                    'SCEC': {'SCE': -0.387241},
                                    'SCEN': {'SCE': -0.093452},
                                    'SCNW': {'SCE': -0.074549},
                                    'SCHD': {'SCE': -0.052128},
                                    'SCLD': {'SCE': -0.002921},
                                    'SDG1': {'SDGE': -1}
                                    },
                                }

zonal_demand_weightings_dict = {'ISONE': {'isone': {'isone': 1},
                                         'HQ': {'HQ': 1},
                                    },
                                'NYISO': {'NYZA': {'NYZA': 1},
                                        'NYZB': {'NYZB': 1},
                                        'NYZC': {'NYZC': 1},
                                        'NYZD': {'NYZD': 1},
                                        'NYZE': {'NYZE': 1},
                                        'NYZF': {'NYZF': 1},
                                        'NYZG': {'NYZG': 1},
                                        'NYZH': {'NYZH': 1},
                                        'NYZI': {'NYZI': 1},
                                        'NYZJ': {'NYZJ': 1},
                                        'NYZK': {'NYZK': 1},
                                    },
                                'PJM': {'AE': {'MidA': 0.038142239},
                                        'AEP': {'West': 0.402775193},
                                        'APS': {'West': 0.151667885},
                                        'ATSI': {'West': 0.220650219},
                                        'BGE': {'MidA': 0.115607756},
                                        'COMED': {'COMED': 1},
                                        'DAYTON': {'West': 0.05425052},
                                        'DEOK': {'West': 0.084539332},
                                        'DOM': {'South': 1},
                                        'DPL': {'MidA': 0.066899284},
                                        'DUQ': {'West': 0.04546856},
                                        'EKPC': {'West': 0.040648291},
                                        'JCPL': {'MidA': 0.084460937},
                                        'METED': {'MidA': 0.054785517},
                                        'PE': {'MidA': 0.146289344},
                                        'PN': {'MidA': 0.066447217},
                                        'PEPCO': {'MidA': 0.110962773},
                                        'PPL': {'MidA': 0.149842508},
                                        'PSEG': {'MidA': 0.16067545},
                                        'RECO': {'MidA': 0.005886975},
                                    },
                                'MISO': {'MISO_AR': {'SOUTH': 0.217990646},
                                        'MISO_DEIN': {'CENTRAL': 0.139430672},
                                        'MISO_HE': {'CENTRAL': 0.00727192},
                                        'MISO_IL': {'CENTRAL': 0.137059279},
                                        'MISO_IPL': {'CENTRAL': 0.036557512},
                                        'MISO_ITC_DTE': {'CENTRAL': 0.157254519},
                                        'MISO_MDU': {'CENTRAL': 0.028971561},
                                        'MISO_METC_CONS': {'CENTRAL': 0.125106095},
                                        'MISO_MO': {'CENTRAL': 0.116891446},
                                        'MISO_MS': {'SOUTH': 0.07774284},
                                        'MISO_NIPS': {'CENTRAL': 0.044779674},
                                        'MISO_SIGE': {'CENTRAL': 0.026551663},
                                        'MISO_TXLA': {'SOUTH': 0.704266514},
                                        'MRO_IOWA': {'NORTH': 0.338286706},
                                        'MRO_MINN': {'NORTH': 0.632741733},
                                        'MRO_WUMS': {'CENTRAL': 0.012995962},
                                        'RFC_WUMS': {'CENTRAL': 0.174237538},
                                        'SERC_BREC': {'CENTRAL': 0.02186372},
                                    },
                                'SPP': {'AE': {'MidA': 0.038142239},
                                    'AEP': {'West': 0.402775193},
                                    'APS': {'West': 0.151667885},
                                    },
                                'ERCOT': {'North_C': {'North_C': 1},
                                         'South_C': {'South_C': 1},
                                         'Coast': {'Coast': 1},
                                         'Southern': {'Southern': 1},
                                         'North': {'North': 1},
                                         'West': {'West': 1},
                                         'Far_West': {'Far_West': 1},
                                         'East': {'East': 1},
                                    },
                                'CAISO': {'PGEB': {'PGE': 0.169364},
                                    'PGSB': {'PGE': 0.142895},
                                    'PGF1': {'PGE': 0.12053},
                                    'PGNP': {'PGE': 0.088599},
                                    'PGZP': {'PGE': 0.074663},
                                    'PGSI': {'PGE': 0.070446},
                                    'PGSF': {'PGE': 0.067711},
                                    'PGP2': {'PGE': 0.059983},
                                    'PGST': {'PGE': 0.044281},
                                    'PGCC': {'PGE': 0.040074},
                                    'PGKN': {'PGE': 0.038247},
                                    'PGFG': {'PGE': 0.03286},
                                    'PGNB': {'PGE': 0.030126},
                                    'PGNC': {'PGE': 0.010781},
                                    'PGHB': {'PGE': 0.009442},
                                    'SCEW': {'SCE': 0.389709},
                                    'SCEC': {'SCE': 0.387241},
                                    'SCEN': {'SCE': 0.093452},
                                    'SCNW': {'SCE': 0.074549},
                                    'SCHD': {'SCE': 0.052128},
                                    'SCLD': {'SCE': 0.002921},
                                    'SDG1': {'SDGE': 1}
                                    },
                            }



constraint_limit_dict = {'ISONE': {'no_constraints': 1
                                    },
                        'NYISO': {'ADIRNDCK 230 MOSES    230 1': 132,
                                    'BUFALO78 115 HUNTLEY  115 1': 94,
                                    'CENTRAL EAST - VC': 2500,
                                    'DUNWODIE 345 SHORE_RD 345 1': 600,
                                    'MOSES SOUTH': 2300,
                                    'NIAGB130 115 PACKARD  115 1': 132
                                    },
                        'PJM': {   'Bagley - Graceton 230kV': 1200,
                                   'Chicago - Praxair3 138kV': 350,
                                   'Harwood - Susquehanna 230kV': 513,
                                   'Logtown - N Delphos 138kV': 58,
                                   'Loretto - Vienna 138kV': 28,
                                   'Messick Rd - Ridgely 138kV': 224,
                                   'TMI 500kV XF': 3200,
                                   'AP South Interface': 3000,
                                   'Smithton - Yukon 138kV': 287,
                                   'Plymouth - Whitpain 230kV': 218,
                                   'Yukon 500kV XF': 491,
                                   'Bellefonte 138kV XF': 255,
                                   'Batesville - Hubble 138kV': 58,
                                   'Monroe - Vineland 69kV': 138,
                                   'Edgewood - Shelocta 115kV': 145
                                   },
                        'MISO': {'Rochester - Wabaco 161 kV': 227,
                                   'South Bend XF 161/115 kV': 187,
                                   'Grand Mound - Maquoketa': 223,
                                   'Sandburg XF 138/161 kV': 300,
                                   'NW Tap - Purdue 138 kV': 190,
                                   'Paradise - BR Tap 161 kV': 455,
                                   'S-N Power Balance': 1900,
                                   'N-S Power Balance': 3000,
                                   'Cheetah - Hot Springs 115 kV': 260,
                                   'Tahlequah - Highway 59 161 kV': 148,
                                   'Coughlin - Manuel 138 kV': 267,
                                   'Dumas - Reed 115 kV': 120,
                                   'Tallulah - Delhi 115 kV': 79,
                                   'Delhi - Tallulah 115 kV': 79,
                                   'LN 587 - Jeffcon 138 kV': 151,
                                   'Bullock - Gleaner 138 kV': 200,
                                   'Gleanor - Bullock 138 kV': 200,
                                  },
                        'SPP': {'no_constraints': 1
                                },
                        'ERCOT': {'NE_LOB': 1021
                                },
                        'CAISO': {'30900_GATES   _230_30970_MIDWAY  _230_BR_1 _1': 2146,
                                '30763_Q0577SS _230_30765_LOSBANOS_230_BR_1 _1 PG1': 2182,
                                '30055_GATES1  _500_30060_MIDWAY  _500_BR_1 _3': 2837,
                                '30060_MIDWAY  _500_24156_VINCENT _500_BR_1 _3': 2146,
                                '30060_MIDWAY  _500_24156_VINCENT _500_BR_2 _3': 2182,
                                '30060_MIDWAY  _500_29402_WIRLWIND_500_BR_1 _1': 2837,
                                '30060_MIDWAY  _500_24156_VINCENT _500_BR_1 _3': 2146,
                                '30060_MIDWAY  _500_24156_VINCENT _500_BR_2 _3': 2182,
                                '30060_MIDWAY  _500_29402_WIRLWIND_500_BR_1 _1': 2837,
                                },
                        }



def get_constraint_sf_dict(agg_constraint_sf, ISO):
    constraint_sf_dict = {}
    #iterate over the tuples where 'ISO' equals the parameter ISO
    for row in agg_constraint_sf[agg_constraint_sf['ISO'] == ISO].itertuples():
        if row.ISO not in constraint_sf_dict.keys():
            constraint_sf_dict[row.ISO] = {}
        if row.constraint not in constraint_sf_dict[row.ISO].keys():
            constraint_sf_dict[row.ISO][row.constraint] = {}
            constraint_sf_dict[row.ISO][row.constraint]['zones'] = {}
            constraint_sf_dict[row.ISO][row.constraint]['hubs'] = {}
        if row.zone not in constraint_sf_dict[row.ISO][row.constraint]['zones'].keys() and row.zone not in iso_hub_dict[ISO]:
            constraint_sf_dict[row.ISO][row.constraint]['zones'][row.zone] = row.shift_factor
        if row.zone not in constraint_sf_dict[row.ISO][row.constraint]['zones'].keys() and ISO == 'NYISO' and row.zone in iso_hub_dict[ISO]:
            constraint_sf_dict[row.ISO][row.constraint]['zones'][row.zone] = row.shift_factor
        if row.zone not in constraint_sf_dict[row.ISO][row.constraint]['hubs'].keys() and row.zone in iso_hub_dict[ISO]:
            constraint_sf_dict[row.ISO][row.constraint]['hubs'][row.zone] = row.shift_factor
    
    return constraint_sf_dict

def queryDB(query, database, ISO):
    conn = pyodbc.connect(str('driver={SQL Server};'+db_credential_dict[database].format(SEER_ISO = ISO)))
    cursor = conn.cursor()
    data = pd.read_sql(query, conn)
    conn.close()

    return (data)
