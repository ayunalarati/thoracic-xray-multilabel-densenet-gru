"""
Configuration file for Multi-Label Thoracic Disease Classification on Chest X-Ray Images.
"""

CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

THRESHOLD = 0.20

AUROC_SCORES = {
    'Atelectasis': 0.7028,
    'Cardiomegaly': 0.7196,
    'Consolidation': 0.7628,
    'Edema': 0.7925,
    'Effusion': 0.7933,
    'Emphysema': 0.7397,
    'Fibrosis': 0.6917,
    'Hernia': 0.8540,
    'Infiltration': 0.6068,
    'Mass': 0.6229,
    'No Finding': 0.6974,
    'Nodule': 0.5786,
    'Pleural_Thickening': 0.6841,
    'Pneumonia': 0.6607,
    'Pneumothorax': 0.7603
}
