from aenum import Enum, skip
import numpy as np


class D(Enum):
    @skip
    class NormalizationArgs(Enum):
        mean__aligned_full = 0.7413874864578247
        std__aligned_full = 0.43787217140197754
        mean__aligned_residual_template = -0.022352544590830803
        std__aligned_residual_template = 0.23946423828601837
        mean__aligned_residual_global_average = 1.3518193675565726e-08
        std__aligned_residual_global_average = 0.2002917230129242
        
        mean__full = None
        std__full = None
        mean__full_residual_global_average = None
        std__full_residual_global_average = None
        
    @skip
    class NormalImageBinarizationThresholds(Enum):
        min_black_pixel_count = 850
        max_black_pixel_count = 1300
    
    @skip
    class DatasetArgs(Enum):
        # manually filtered book list based off character skeleton height and # char >= 5
        book_list = None

class F(Enum):
    @skip
    class NormalizationArgs(Enum):
        mean__aligned_full = 0.8050340414047241
        std__aligned_full = 0.3961746394634247
        mean__aligned_residual_template = -0.013342983089387417
        std__aligned_residual_template = 0.248682901263237
        mean__aligned_residual_global_average = 5.2511652803843845e-09
        std__aligned_residual_global_average = 0.21090006828308105
        
        mean__full = None
        std__full = None
        mean__full_residual_global_average = None
        std__full_residual_global_average = None
        
    @skip
    class NormalImageBinarizationThresholds(Enum):
        min_black_pixel_count = 650
        max_black_pixel_count = 1000
    
    @skip
    class DatasetArgs(Enum):
        # manually filtered book list based off character skeleton height and # char >= 5
        book_list = None


class G(Enum):
    @skip
    class NormalizationArgs(Enum):
        mean__aligned_full = 0.7795137763023376
        std__aligned_full = 0.4145748019218445
        mean__aligned_residual_template = -0.015499626286327839
        std__aligned_residual_template = 0.2561003565788269
        mean__aligned_residual_global_average = -5.3833282720461284e-08
        std__aligned_residual_global_average = 0.22370268404483795
        
        mean__full = 0.72342277
        std__full = 0.44730595
        mean__full_residual_global_average = None
        std__full_residual_global_average = None
        
    @skip
    class ItalicsDetectorArgs(Enum):
        # for italic detector
        mean__full = 0.5680886507034302
        std__full = 0.11926612257957458
        mean__intensities_per_channel = (156, 142, 126)
#         max_h = 150
#         max_w = 150
        max_h = 64
        max_w = 64
        
    @skip
    class NormalImageBinarizationThresholds(Enum):
        min_black_pixel_count = 500
        max_black_pixel_count = 1200
    
    @skip
    class DatasetArgs(Enum):
        # manually filtered book list based off character skeleton height and # char >= 5
        book_list = None
        

class M(Enum):
    @skip
    class NormalizationArgs(Enum):
        mean__aligned_full = 0.7089159488677979
        std__aligned_full = 0.4542621374130249
        mean__aligned_residual_template = -0.04593479260802269
        std__aligned_residual_template = 0.2924497127532959
        mean__aligned_residual_global_average = 3.5561566136266265e-08
        std__aligned_residual_global_average = 0.2427058219909668
        
        mean__full = None
        std__full = None
        mean__full_residual_global_average = None
        std__full_residual_global_average = None
    
    @skip
    class NormalImageBinarizationThresholds(Enum):
        min_black_pixel_count = 800
        max_black_pixel_count = 1600
        
    
    @skip
    class DatasetArgs(Enum):
        # manually filtered book list based off character skeleton height and #char >= 5
        book_list = None
        

def get_char_settings(char):
    # from perturber_args.py
    if char == 'G':
        return G
    elif char == 'M':
        return M
    elif char == 'D':
        return D
    elif char == 'F':
        return F
    # cover other characters using these settings
    elif char in set("ET"):
        return F
    elif char in set("JPR"):
        return D
    elif char in set("BCQS"):
        return G
    elif char in set("AHKLNUVWXYZ"):
        return M
    else:
        raise ValueError(f'Char {char} does not have a settings Enum.')
