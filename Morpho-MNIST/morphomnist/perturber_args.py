from aenum import Enum, skip
import numpy as np


# "skeletonize" or "medial_axis"
SKELETONIZER = "medial_axis"
DEBUG = False

FRACTURE_erosion_dilation_amount__lower = 25.
FRACTURE_erosion_dilation_amount__upper = 35.
FRACTURE_erosion_dilation_amount__mu = 30.
FRACTURE_erosion_dilation_amount__sigma = 1.

class D(Enum):
    @skip
    class BendArgs(Enum):
        char = 'D'
        distance__lower = 15+15
        distance__upper = 25+15
        distance__mu = 20+15
        distance__sigma = 2.5

        dist_shift__lower = 4+4
        dist_shift__upper = 10+4
        dist_shift__mu = 7+4
        dist_shift__sigma = 1.5

        prune_tips = 3
        prune_forks = 4
        skeletonizer = SKELETONIZER  
        relative = True
        interpolation_bending = False
        midpoints = None
        angle_seed = 0
        debug = DEBUG
    
#     @skip - original args before using two base images
#     class BendArgs(Enum):
#         distance__lower = 15
#         distance__upper = 25
#         distance__mu = 20
#         distance__sigma = 2.5

#         dist_shift__lower = 4
#         dist_shift__upper = 10
#         dist_shift__mu = 7
#         dist_shift__sigma = 1.5

#         prune_tips = 3
#         prune_forks = 7
#         skeletonizer = SKELETONIZER  
#         relative = True
#         interpolation_bending = False
#         midpoints = None
#         angle_seed = 0
#         debug = DEBUG

    @skip
    class FractureArgs(Enum):
        angle__mu = np.pi / 2
        angle__sigma = np.pi / 12
        angle__lower = np.pi / 6
        angle__upper = 5 * np.pi / 6
        
        frac_extension = 3.
        prune = 3
        num_frac = 1
        
        thickness__lower = 55.
        thickness__upper = 75.
        thickness__mu = 65.
        thickness__sigma = 5.

        # erosion_dilation_amount__lower = 20.
        # erosion_dilation_amount__upper = 30.
        # erosion_dilation_amount__mu = 25.
        # erosion_dilation_amount__sigma = 2.
        erosion_dilation_amount__lower = FRACTURE_erosion_dilation_amount__lower
        erosion_dilation_amount__upper = FRACTURE_erosion_dilation_amount__upper
        erosion_dilation_amount__mu = FRACTURE_erosion_dilation_amount__mu
        erosion_dilation_amount__sigma = FRACTURE_erosion_dilation_amount__sigma
        debug = DEBUG
        
    
    @skip
    class ThinningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
  
    @skip
    class ThickeningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
        
    # @skip
    # class SwellingArgs(Enum):
    #     p_swell = 0.5
    #     strength__lower = 1
    #     strength__upper = 4
    #     strength__mu = 2.5
    #     strength__sigma = 0.5
    #     radius__lower = 4
    #     radius__upper = 7
    #     radius__mu = 5.5
    #     radius__sigma = 0.5
    @skip
    class SwellingArgs(Enum):
        p_swell = 0.5
        strength__lower = 1
        strength__upper = 7
        strength__mu = 4
        strength__sigma = 1.0
        radius__lower = 4
        radius__upper = 10
        radius__mu = 7
        radius__sigma = 1.0


class F(Enum):
    @skip
    class BendArgs(Enum):
        char = 'F'
        distance__lower = 15+15
        distance__upper = 25+15
        distance__mu = 20+15
        distance__sigma = 2.5 + 3

        dist_shift__lower = 4+5
        dist_shift__upper = 10+5
        dist_shift__mu = 7+5
        dist_shift__sigma = 1.5 + 2

        prune_tips = 2
        prune_forks = 2
        skeletonizer = SKELETONIZER  
        relative = True
        interpolation_bending = False
        midpoints = None
        angle_seed = 0
        debug = DEBUG

    @skip
    class FractureArgs(Enum):
        angle__mu = np.pi / 2
        angle__sigma = np.pi / 12
        angle__lower = np.pi / 6
        angle__upper = 5 * np.pi / 6
        
        frac_extension = 2.0
        prune = 2
        num_frac = 1
        
        thickness__lower = 45.
        thickness__upper = 70.
        thickness__mu = 57.
        thickness__sigma = 8.

        # erosion_dilation_amount__lower = 20.
        # erosion_dilation_amount__upper = 30.
        # erosion_dilation_amount__mu = 25.
        # erosion_dilation_amount__sigma = 2.
        erosion_dilation_amount__lower = FRACTURE_erosion_dilation_amount__lower
        erosion_dilation_amount__upper = FRACTURE_erosion_dilation_amount__upper
        erosion_dilation_amount__mu = FRACTURE_erosion_dilation_amount__mu
        erosion_dilation_amount__sigma = FRACTURE_erosion_dilation_amount__sigma
        debug = DEBUG
        
    @skip
    class ThinningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
  
    @skip
    class ThickeningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
        
    # @skip
    # class SwellingArgs(Enum):
    #     p_swell = 0.5
    #     strength__lower = 1
    #     strength__upper = 4
    #     strength__mu = 2.5
    #     strength__sigma = 0.5
    #     radius__lower = 4
    #     radius__upper = 7
    #     radius__mu = 5.5
    #     radius__sigma = 0.5
    @skip
    class SwellingArgs(Enum):
        p_swell = 0.5
        strength__lower = 1
        strength__upper = 7
        strength__mu = 4
        strength__sigma = 1.0
        radius__lower = 4
        radius__upper = 10
        radius__mu = 7
        radius__sigma = 1.0


class G(Enum):
    @skip
    class BendArgs(Enum):
        char = 'G'
        # approx_distance > 35 is generally a good, noticeable bend
        distance__lower = 35
        distance__upper = 49
        distance__mu = 42
        distance__sigma = 2.5

        dist_shift__lower = 8
        dist_shift__upper = 20
        dist_shift__mu = 16
        dist_shift__sigma = 3

        prune_tips = 3
        prune_forks = 4
        skeletonizer = SKELETONIZER  
        relative = True
        interpolation_bending = False
        midpoints = None
        angle_seed = 0
        debug = DEBUG

    @skip
    class FractureArgs(Enum):
        angle__mu = np.pi / 2
        angle__sigma = np.pi / 12
        angle__lower = np.pi / 6
        angle__upper = 5 * np.pi / 6
        
        frac_extension = 3.
        prune = 6
        num_frac = 1
        
        thickness__lower = 50.
        thickness__upper = 70.
        thickness__mu = 57.
        thickness__sigma = 5.
        # erosion_dilation_amount__lower = 20.
        # erosion_dilation_amount__upper = 30.
        # erosion_dilation_amount__mu = 25.
        # erosion_dilation_amount__sigma = 2.
        erosion_dilation_amount__lower = FRACTURE_erosion_dilation_amount__lower
        erosion_dilation_amount__upper = FRACTURE_erosion_dilation_amount__upper
        erosion_dilation_amount__mu = FRACTURE_erosion_dilation_amount__mu
        erosion_dilation_amount__sigma = FRACTURE_erosion_dilation_amount__sigma
        
        debug = DEBUG
    
    @skip
    class ThinningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.075
  
    @skip
    class ThickeningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
        
    # @skip
    # class SwellingArgs(Enum):
    #     p_swell = 0.5
    #     strength__lower = 1
    #     strength__upper = 4
    #     strength__mu = 2.5
    #     strength__sigma = 0.5
    #     radius__lower = 4
    #     radius__upper = 7
    #     radius__mu = 5.5
    #     radius__sigma = 0.5
    @skip
    class SwellingArgs(Enum):
        p_swell = 0.5
        strength__lower = 1
        strength__upper = 7
        strength__mu = 4
        strength__sigma = 1.0
        radius__lower = 4
        radius__upper = 10
        radius__mu = 7
        radius__sigma = 1.0


class M(Enum):
    @skip
    class BendArgs(Enum):
        char = 'M'
        distance__lower = 24
        distance__upper = 49
        distance__mu = 40
        distance__sigma = 9

        dist_shift__lower = 13
        dist_shift__upper = 28
        dist_shift__mu = 20
        dist_shift__sigma = 2

        prune_tips = 3
        prune_forks = 3
        skeletonizer = SKELETONIZER  
        relative = True
        interpolation_bending = False
        midpoints = None
        angle_seed = 0
        
        debug = DEBUG

    @skip
    class FractureArgs(Enum):
        angle__mu = np.pi / 2
        angle__sigma = np.pi / 12
        angle__lower = np.pi / 6
        angle__upper = 5 * np.pi / 6
        
        frac_extension = 2.0
        prune = 3
        num_frac = 1
        
        thickness__lower = 50.
        thickness__upper = 70.
        thickness__mu = 57.
        thickness__sigma = 5.
        # erosion_dilation_amount__lower = 20.
        # erosion_dilation_amount__upper = 30.
        # erosion_dilation_amount__mu = 25.
        # erosion_dilation_amount__sigma = 2.
        erosion_dilation_amount__lower = FRACTURE_erosion_dilation_amount__lower
        erosion_dilation_amount__upper = FRACTURE_erosion_dilation_amount__upper
        erosion_dilation_amount__mu = FRACTURE_erosion_dilation_amount__mu
        erosion_dilation_amount__sigma = FRACTURE_erosion_dilation_amount__sigma
        
        debug = DEBUG
    
    @skip
    class ThinningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
  
    @skip
    class ThickeningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
        
    # @skip
    # class SwellingArgs(Enum):
    #     p_swell = 0.5
    #     strength__lower = 1
    #     strength__upper = 4
    #     strength__mu = 2.5
    #     strength__sigma = 0.5
    #     radius__lower = 4
    #     radius__upper = 7
    #     radius__mu = 5.5
    #     radius__sigma = 0.5
    @skip
    class SwellingArgs(Enum):
        p_swell = 0.5
        strength__lower = 1
        strength__upper = 7
        strength__mu = 4
        strength__sigma = 1.0
        radius__lower = 4
        radius__upper = 10
        radius__mu = 7
        radius__sigma = 1.0


class All(Enum):
    @skip
    class BendArgs(Enum):
        distance__lower = 35
        distance__upper = 49
        distance__mu = 42
        distance__sigma = 2.5

        dist_shift__lower = 8
        dist_shift__upper = 20
        dist_shift__mu = 16
        dist_shift__sigma = 3

        prune_tips = 3
        prune_forks = 4
        skeletonizer = SKELETONIZER  
        relative = True
        interpolation_bending = False
        midpoints = None
        angle_seed = 0
        debug = DEBUG

    @skip
    class FractureArgs(Enum):
        angle__mu = np.pi / 2
        angle__sigma = np.pi / 12
        angle__lower = np.pi / 6
        angle__upper = 5 * np.pi / 6
        
        frac_extension = 3.
        prune = 6
        num_frac = 1
        
        thickness__lower = 60.
        thickness__upper = 80.
        thickness__mu = 70.
        thickness__sigma = 5.

        erosion_dilation_amount__lower = FRACTURE_erosion_dilation_amount__lower
        erosion_dilation_amount__upper = FRACTURE_erosion_dilation_amount__upper
        erosion_dilation_amount__mu = FRACTURE_erosion_dilation_amount__mu
        erosion_dilation_amount__sigma = FRACTURE_erosion_dilation_amount__sigma
        debug = DEBUG

    @skip
    class ThinningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.075
  
    @skip
    class ThickeningArgs(Enum):
        amount__lower = 0.1
        amount__upper = 0.4
        amount__mu = 0.25
        amount__sigma = 0.15
        
    @skip
    class SwellingArgs(Enum):
        p_swell = 0.5
        strength__lower = 1
        strength__upper = 7
        strength__mu = 4
        strength__sigma = 1.0
        radius__lower = 4
        radius__upper = 10
        radius__mu = 7
        radius__sigma = 1.0


def get_char_settings(char):
    return All
    # alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W']
    # ABCDEFGHJKLMNPRSTUVW
    # if char == 'G':
    #     return G
    # elif char == 'M':
    #     return M
    # elif char == 'D':
    #     return D
    # elif char == 'F':
    #     return F
    # # cover other characters using these settings
    # elif char in set("EILTY"):
    #     return F
    # elif char in set("JOPQR"):
    #     return D
    # elif char in set("CSU"):
    #     return G
    # elif char in set("ABHKNVWXZ"):
    #     return M
    # else:
    #     raise ValueError(f'Char {char} does not have a settings Enum.')


if __name__=="__main__":
    pass
    # example usage: G.BendArgs.distance.value 
