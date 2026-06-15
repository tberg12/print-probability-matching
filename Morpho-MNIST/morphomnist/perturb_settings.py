from morphomnist import perturb
import numpy as np
from scipy.stats import truncnorm

import random


class TwinSettings:
    def __init__(self, char_settings):
        """ Perturb settings for the Twin data generation. This differs from 
        the AnomalyDetection because we want more variation in some areas
        so we use a uniform dist. instead of trunc. norm. in some places..
        
        Args:
            char_settings: a character Enum object from perturber_args
        """
        self.char_settings = char_settings
        
        #
        #    GLOBAL INKING
        #
        
        self.global_inking_perturbations = (
            lambda: perturb.Identity(),
#             lambda: perturb.Thinning(amount=np.random.uniform(char_settings.ThinningArgs.amount__lower.value, 
#                                                               char_settings.ThinningArgs.amount__upper.value)),
#             lambda: perturb.Thickening(amount=np.random.uniform(char_settings.ThickeningArgs.amount__lower.value, 
#                                                                 char_settings.ThickeningArgs.amount__upper.value)),
            # was (0.2, 0.4)
            lambda: perturb.Thinning(amount=np.random.uniform(0.1, 
                                                              0.35)),
            lambda: perturb.Thickening(amount=np.random.uniform(0.075, 
                                                                0.235)),
        )
        
        # NOTE:
        self.global_inking_perturbations_probs = (
            0.05,
            0.475,
            0.475
        )
        
        #
        #    LOCAL INKING
        #
        
        self.local_inking_perturbation = lambda: perturb.Swelling(
            strength=truncnorm.rvs(
                (char_settings.SwellingArgs.strength__lower.value - char_settings.SwellingArgs.strength__mu.value) / char_settings.SwellingArgs.strength__sigma.value, 
                (char_settings.SwellingArgs.strength__upper.value - char_settings.SwellingArgs.strength__mu.value) / char_settings.SwellingArgs.strength__sigma.value, 
                loc=char_settings.SwellingArgs.strength__mu.value, 
                scale=char_settings.SwellingArgs.strength__sigma.value
            ), 
            radius=truncnorm.rvs(
                (char_settings.SwellingArgs.radius__lower.value - char_settings.SwellingArgs.radius__mu.value) / char_settings.SwellingArgs.radius__sigma.value, 
                (char_settings.SwellingArgs.radius__upper.value - char_settings.SwellingArgs.radius__mu.value) / char_settings.SwellingArgs.radius__sigma.value, 
                loc=char_settings.SwellingArgs.radius__mu.value, 
                scale=char_settings.SwellingArgs.radius__sigma.value
            )
        ) if random.random() < char_settings.SwellingArgs.p_swell.value else perturb.Identity()



        #
        #    DAMAGE
        #
        
        self.damage_perturbations = (
            lambda: perturb.PairBend(
                                 distance = truncnorm.rvs(
                                                    (char_settings.BendArgs.distance__lower.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    (char_settings.BendArgs.distance__upper.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    loc=char_settings.BendArgs.distance__mu.value, 
                                                    scale=char_settings.BendArgs.distance__sigma.value,
                                        ).astype(np.int32),
                                 dist_shift = truncnorm.rvs(
                                                    (char_settings.BendArgs.dist_shift__lower.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    (char_settings.BendArgs.dist_shift__upper.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    loc=char_settings.BendArgs.dist_shift__mu.value, 
                                                    scale=char_settings.BendArgs.dist_shift__sigma.value,
                                        ).astype(np.int32),
                                 prune_tips=char_settings.BendArgs.prune_tips.value,
                                 prune_forks=char_settings.BendArgs.prune_forks.value,
                                 skeletonizer=char_settings.BendArgs.skeletonizer.value,
                                 relative = char_settings.BendArgs.relative.value,
                                 interpolation_bending = char_settings.BendArgs.interpolation_bending.value,
                                 char = char_settings.BendArgs.char.value,
                                 debug = char_settings.BendArgs.debug.value
                                ),
            lambda: perturb.PairOpenFracture(num_frac=char_settings.FractureArgs.num_frac.value, 
                                        angle_to_skeleton=truncnorm.rvs(
                                            (char_settings.FractureArgs.angle__lower.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            (char_settings.FractureArgs.angle__upper.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            loc=char_settings.FractureArgs.angle__mu.value, 
                                            scale=char_settings.FractureArgs.angle__sigma.value
                                        ),
                                        thickness=truncnorm.rvs(
                                            (char_settings.FractureArgs.thickness__lower.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            (char_settings.FractureArgs.thickness__upper.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            loc=char_settings.FractureArgs.thickness__mu.value, 
                                            scale=char_settings.FractureArgs.thickness__sigma.value
                                        ),
                                        erosion_dilation_amount=truncnorm.rvs(
                                                    (char_settings.FractureArgs.erosion_dilation_amount__lower.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    (char_settings.FractureArgs.erosion_dilation_amount__upper.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    loc=char_settings.FractureArgs.erosion_dilation_amount__mu.value, 
                                                    scale=char_settings.FractureArgs.erosion_dilation_amount__sigma.value,
                                        ),
                                        frac_extension=char_settings.FractureArgs.frac_extension.value,
                                        prune=char_settings.FractureArgs.prune.value
                                    ),

        )
        self.damage_perturbatinos_pair = self.damage_perturbations
        # also define the single perturbations for the single images (same base image)
        self.damage_perturbations_single = (
            lambda: perturb.SingleBend(
                                 distance = truncnorm.rvs(
                                                    (char_settings.BendArgs.distance__lower.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    (char_settings.BendArgs.distance__upper.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    loc=char_settings.BendArgs.distance__mu.value, 
                                                    scale=char_settings.BendArgs.distance__sigma.value,
                                        ).astype(np.int32),
                                 dist_shift = truncnorm.rvs(
                                                    (char_settings.BendArgs.dist_shift__lower.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    (char_settings.BendArgs.dist_shift__upper.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    loc=char_settings.BendArgs.dist_shift__mu.value, 
                                                    scale=char_settings.BendArgs.dist_shift__sigma.value,
                                        ).astype(np.int32),
                                 prune_tips=char_settings.BendArgs.prune_tips.value,
                                 prune_forks=char_settings.BendArgs.prune_forks.value,
                                 skeletonizer=char_settings.BendArgs.skeletonizer.value,
                                 relative = char_settings.BendArgs.relative.value,
                                 interpolation_bending = char_settings.BendArgs.interpolation_bending.value,
                                 char = char_settings.BendArgs.char.value,
                                 debug = char_settings.BendArgs.debug.value
                                ),
            lambda: perturb.OpenFracture(num_frac=char_settings.FractureArgs.num_frac.value, 
                                        angle_to_skeleton=truncnorm.rvs(
                                            (char_settings.FractureArgs.angle__lower.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            (char_settings.FractureArgs.angle__upper.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            loc=char_settings.FractureArgs.angle__mu.value, 
                                            scale=char_settings.FractureArgs.angle__sigma.value
                                        ),
                                        thickness=truncnorm.rvs(
                                            (char_settings.FractureArgs.thickness__lower.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            (char_settings.FractureArgs.thickness__upper.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            loc=char_settings.FractureArgs.thickness__mu.value, 
                                            scale=char_settings.FractureArgs.thickness__sigma.value
                                        ),
                                        erosion_dilation_amount=truncnorm.rvs(
                                                    (char_settings.FractureArgs.erosion_dilation_amount__lower.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    (char_settings.FractureArgs.erosion_dilation_amount__upper.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    loc=char_settings.FractureArgs.erosion_dilation_amount__mu.value, 
                                                    scale=char_settings.FractureArgs.erosion_dilation_amount__sigma.value,
                                        ),
                                        frac_extension=char_settings.FractureArgs.frac_extension.value,
                                        prune=char_settings.FractureArgs.prune.value
                                    ),

        )


        #
        #    WARPING (TRANSLATION, SCALE, SHEAR)
        #
        
        # TODO:
        
        random_rotation_angle = 0.
        random_translate = 0.
        random_scale = 0.
        random_shear = 0.
    
    
class AnomalyDetectionSettings:
    def __init__(self, char_settings):
        """ Perturb settings for the Anomaly Detection data augmentation.
        
        Args:
            char_settings: a character Enum object from perturber_args
        """
        self.char_settings = char_settings
        
        #
        #    GLOBAL INKING
        #
        
        self.global_inking_perturbations = (
            lambda: perturb.Identity(),
            lambda: perturb.Thinning(amount=truncnorm.rvs(
                (char_settings.ThinningArgs.amount__lower.value - char_settings.ThinningArgs.amount__mu.value) / char_settings.ThinningArgs.amount__sigma.value, 
                (char_settings.ThinningArgs.amount__upper.value - char_settings.ThinningArgs.amount__mu.value) / char_settings.ThinningArgs.amount__sigma.value, 
                loc=char_settings.ThinningArgs.amount__mu.value, 
                scale=char_settings.ThinningArgs.amount__sigma.value), 
            ),
            lambda: perturb.Thickening(amount=truncnorm.rvs(
                (char_settings.ThickeningArgs.amount__lower.value - char_settings.ThickeningArgs.amount__mu.value) / char_settings.ThickeningArgs.amount__sigma.value, 
                (char_settings.ThickeningArgs.amount__upper.value - char_settings.ThickeningArgs.amount__mu.value) / char_settings.ThickeningArgs.amount__sigma.value, 
                loc=char_settings.ThickeningArgs.amount__mu.value, 
                scale=char_settings.ThickeningArgs.amount__sigma.value), 
            ),
        )
        
        self.global_inking_perturbations_probs = (
            0.03,
            0.485,
            0.485
        )
        
        #
        #    LOCAL INKING
        #
        
        self.local_inking_perturbation = lambda: perturb.Swelling(
            strength=truncnorm.rvs(
                (char_settings.SwellingArgs.strength__lower.value - char_settings.SwellingArgs.strength__mu.value) / char_settings.SwellingArgs.strength__sigma.value, 
                (char_settings.SwellingArgs.strength__upper.value - char_settings.SwellingArgs.strength__mu.value) / char_settings.SwellingArgs.strength__sigma.value, 
                loc=char_settings.SwellingArgs.strength__mu.value, 
                scale=char_settings.SwellingArgs.strength__sigma.value
            ), 
            radius=truncnorm.rvs(
                (char_settings.SwellingArgs.radius__lower.value - char_settings.SwellingArgs.radius__mu.value) / char_settings.SwellingArgs.radius__sigma.value, 
                (char_settings.SwellingArgs.radius__upper.value - char_settings.SwellingArgs.radius__mu.value) / char_settings.SwellingArgs.radius__sigma.value, 
                loc=char_settings.SwellingArgs.radius__mu.value, 
                scale=char_settings.SwellingArgs.radius__sigma.value
            )
        ) if random.random() < char_settings.SwellingArgs.p_swell.value else perturb.Identity()

        #
        #    DAMAGE
        #

        self.damage_perturbations = (
            lambda: perturb.Bend(
                                 distance = truncnorm.rvs(
                                                    (char_settings.BendArgs.distance__lower.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    (char_settings.BendArgs.distance__upper.value - char_settings.BendArgs.distance__mu.value) / char_settings.BendArgs.distance__sigma.value, 
                                                    loc=char_settings.BendArgs.distance__mu.value, 
                                                    scale=char_settings.BendArgs.distance__sigma.value,
                                        ).astype(np.int32),
                                 dist_shift = truncnorm.rvs(
                                                    (char_settings.BendArgs.dist_shift__lower.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    (char_settings.BendArgs.dist_shift__upper.value - char_settings.BendArgs.dist_shift__mu.value) / char_settings.BendArgs.dist_shift__sigma.value, 
                                                    loc=char_settings.BendArgs.dist_shift__mu.value, 
                                                    scale=char_settings.BendArgs.dist_shift__sigma.value,
                                        ).astype(np.int32),
                                 prune_tips=char_settings.BendArgs.prune_tips.value,
                                 prune_forks=char_settings.BendArgs.prune_forks.value,
                                 skeletonizer=char_settings.BendArgs.skeletonizer.value,
                                 relative = char_settings.BendArgs.relative.value,
                                 interpolation_bending = char_settings.BendArgs.interpolation_bending.value,
                                 debug = char_settings.BendArgs.debug.value
                                ),
            lambda: perturb.OpenFracture(num_frac=char_settings.FractureArgs.num_frac.value, 
                                        angle_to_skeleton=truncnorm.rvs(
                                            (char_settings.FractureArgs.angle__lower.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            (char_settings.FractureArgs.angle__upper.value - char_settings.FractureArgs.angle__mu.value) / char_settings.FractureArgs.angle__sigma.value, 
                                            loc=char_settings.FractureArgs.angle__mu.value, 
                                            scale=char_settings.FractureArgs.angle__sigma.value
                                        ),
                                        thickness=truncnorm.rvs(
                                            (char_settings.FractureArgs.thickness__lower.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            (char_settings.FractureArgs.thickness__upper.value - char_settings.FractureArgs.thickness__mu.value) / char_settings.FractureArgs.thickness__sigma.value, 
                                            loc=char_settings.FractureArgs.thickness__mu.value, 
                                            scale=char_settings.FractureArgs.thickness__sigma.value
                                        ),
                                        erosion_dilation_amount=truncnorm.rvs(
                                                    (char_settings.FractureArgs.erosion_dilation_amount__lower.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    (char_settings.FractureArgs.erosion_dilation_amount__upper.value - char_settings.FractureArgs.erosion_dilation_amount__mu.value) / char_settings.FractureArgs.erosion_dilation_amount__sigma.value, 
                                                    loc=char_settings.FractureArgs.erosion_dilation_amount__mu.value, 
                                                    scale=char_settings.FractureArgs.erosion_dilation_amount__sigma.value,
                                        ),
                                        frac_extension=char_settings.FractureArgs.frac_extension.value,
                                        prune=char_settings.FractureArgs.prune.value
                                    ),

        )
