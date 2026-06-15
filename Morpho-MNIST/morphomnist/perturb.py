import numpy as np
from skimage import draw, morphology, transform, img_as_bool
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import matplotlib.pyplot as plt

#from . import skeleton
import skeleton
#from .morpho import ImageMorphology
from morpho import ImageMorphology
#


# from . import bender_for_figures as bender
#from . import character_bender as bender # for bend perturbation
import character_bender
# from . import bender_old # for bend perturbation
#from .skeleton import LocationSampler
from skeleton import LocationSampler


def skeleton_nearest_neighbor(point, skel, skel_pixel, debug=False):
    """
        Find the point on binary skeleton nearest to `point`
        point: source point
        skel: skeleton
        skel_pixel: {True, False} - denote if the skeleton pixel is made of white (True) or black (False)
        
        Usage: (x_nn, y_nn) = skeleton_nearest_neighbor((x0, y0), skel, skel_pixel=False, debug=False)
        
    """
    assert set(np.unique(skel)) == {True, False}
    x0, y0 = point
    xs,ys = np.where(skel == skel_pixel)
    distances = [np.linalg.norm([x-x0, y-y0]) for x,y in zip(xs, ys)]
    idx = np.argmin(distances)
    x_nn, y_nn = (xs[idx], ys[idx])
    if debug:
        plt.imshow(skel, cmap="gray")
        plt.plot(y0, x0, '*r', label="src")
        plt.plot(y_nn, x_nn, '*g', label="1NN")
        plt.title(f"{(x0,y0)} -> {x_nn,y_nn}")
        plt.legend()
        plt.show()
    return x_nn, y_nn

class PairPerturbException(Exception): # generic pair bend exception
    pass

class SinglePerturbException(Exception): # generic single bend exception
    pass

class Perturbation:
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        """Apply the perturbation.

        Parameters
        ----------
        morph : ImageMorphology
            Morphological pipeline computed for the input image.

        Returns
        -------
        (scale*H, scale*W) numpy.ndarray
            The perturbed high-resolution image. Call `morph.downscale(...)` to transform it back
            to low-resolution.
        """
        raise NotImplementedError
        
        
class Identity(Perturbation):
    """Do nothing to the image"""

    def __init__(self):
        pass

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        return morph.binary_image


class Thinning(Perturbation):
    """Thin a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = .7, mean_thickness = None):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=0.7` will
            reduce the thickness by approximately 70%).
        """
        self.amount = amount
        self.mean_thickness = mean_thickness

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        if self.mean_thickness is None:
            radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        else:
            radius = int(self.amount * morph.scale * self.mean_thickness / 2.)
        return morphology.erosion(morph.binary_image, morphology.disk(radius))


class Thickening(Perturbation):
    """Thicken a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = 1, mean_thickness = None):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=1.0` will
            increase the thickness by approximately 100%).
        """
        self.amount = amount
        self.mean_thickness = mean_thickness

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        if self.mean_thickness is None:
            radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        else:
            radius = int(self.amount * morph.scale * self.mean_thickness / 2.)
        return morphology.dilation(morph.binary_image, morphology.disk(radius))
    
class BinaryThinning(Perturbation):
    """Thin a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = .7):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=0.7` will
            reduce the thickness by approximately 70%).
        """
        self.amount = amount

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.binary_erosion(morph.binary_image, morphology.disk(radius))


class BinaryThickening(Perturbation):
    """Thicken a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = 1):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=1.0` will
            increase the thickness by approximately 100%).
        """
        self.amount = amount

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.binary_dilation(morph.binary_image, morphology.disk(radius))


class Deformation(Perturbation):
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        warped_img = transform.warp(morph.binary_image, lambda xy: self.warp(xy, morph), clip=True, preserve_range=True)
        assert (warped_img >= 0).all(), f'Negative value encountered in image! {np.unique(warped_img)}'
#         return img_as_bool(warped_img, force_copy=True)
        return warped_img

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        """Transform a regular coordinate grid to the deformed coordinates in input space.

        Parameters
        ----------
        xy : (H*W, 2) numpy.ndarray
            Regular coordinate grid in output space.
        morph : ImageMorphology
            Morphological pipeline computed for the input image.

        Returns
        -------
        (H*W, 2) numpy.ndarray
            Warped coordinates in input space.
        """
        raise NotImplementedError


class Swelling(Deformation):
    """Create a local swelling at a random location along the skeleton.

    Coordinates within `radius` :math:`R` of the centre location :math:`r_0` are warped according
    to a radial power transform: :math:`f(r) = r_0 + (r-r_0)(|r-r_0|/R)^{\gamma-1}`, where
    :math:`\gamma` is the `strength`.
    """

    def __init__(self, strength: float = 3, radius: float = 7):
        """
        Parameters
        ----------
        strength : float, optional
            Exponent of radial power transform (>1).
        radius : float, optional
            Radius to be affected by the swelling, relative to low-resolution pixel scale.
        """
        self.strength = strength
        self.radius = radius
        self.loc_sampler = skeleton.LocationSampler()

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        centre = self.loc_sampler.sample(morph)[::-1]
        radius = (self.radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale

        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (self.strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy


class Fracture(Perturbation):
    """Add fractures to a digit.

    Fractures are added at random locations along the skeleton, while avoiding stroke tips and
    forks, and are locally perpendicular to the pen stroke.
    """

    # these are now parameters in __init__
#     _ANGLE_WINDOW = 2
#     _FRAC_EXTENSION = 2.  # was .5

    def __init__(self, thickness: float = 10., prune: float = 7, num_frac: int = 1, angle_to_skeleton: float = np.pi / 2., centres=None, angle_window: int = 2, frac_extension: float = 2., relative=True):
        """
        Parameters
        ----------
        thickness : float, optional
            Thickness of the fractures, in low-resolution pixel scale.
        prune : float, optional
            Radius to avoid around stroke tips and forks, in low-resolution pixel scale.
        num_frac : int, optional
            Number of fractures to add.
        """
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac
        self.angle_to_skeleton = angle_to_skeleton
        self.loc_sampler = skeleton.LocationSampler(prune, prune)
        self.angle_window = angle_window
        self.frac_extension = frac_extension
        self.centres = centres
        self.relative = relative
    
    def sample_location(self, morph: ImageMorphology):
        """ Uses the LocationSampler on the image's skeleton to get centres.
        """
        try:
            centres = self.loc_sampler.sample(morph, self.num_frac)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, self.num_frac)
        return centres
    
    def set_location(self, centres):
        """ Set the centres attribute of the location.
        """
        self.centres = centres

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        if self.relative:
            y_coords, x_coords = np.where(morph.skeleton > 0)
            y_min, y_max = y_coords.min(), y_coords.max()
            img_h = y_max - y_min
            
            thickness = (self.thickness / 100.) * morph.mean_thickness
            frac_extension = (self.frac_extension / 100.) * img_h
#             print(f'mean_thickness:{morph.mean_thickness},thickness:{thickness},frac_extension:{frac_extension}')
        else:
            thickness = self.thickness
            frac_extension = self.frac_extension

        up_thickness = thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        # create a larger, padded img so that we don't get out of bounds errors when drawing
        frac_img = np.pad(morph.binary_image, pad_width=r, mode='constant', constant_values=False)
        if self.centres is None:
            centres = self.sample_location(morph)
        else:
            centres = self.centres
#         print(f'centres: {centres}')
        for centre in centres:
            p0, p1 = self._endpoints(morph, centre, frac_extension)
#             print(f'Drawing line between p0(={p0}) & p1(={p1})')
            self._draw_line(frac_img, p0, p1, brush)
        # return unpadded img by taking off r from all sides
        return frac_img[r:-r, r:-r]

    def _endpoints(self, morph, centre, frac_extension):
        angle = skeleton.get_angle(morph.skeleton, *centre, self.angle_window * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + frac_extension * morph.scale
        angle += self.angle_to_skeleton  # Perpendicular to the skeleton by default (pi / 2.)
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        # TODO: change h & w to be functions like: h(dist_map, i), w(dist_map, j)
#         import pdb; pdb.set_trace()
#         _, dist_map = morphology.medial_axis(img, return_distance=True)
#         print(f'brush: {brush.shape}, {brush}')
        h, w = brush.shape
#         print(h, w)
        ii, jj = draw.line(*p0, *p1)
#         print(f'ii: {ii.shape}')
#         print(f'jj: {jj.shape}')
#         np.set_printoptions(threshold=10000000)
#         print(f'dist_map: {dist_map.shape}\n{dist_map[42:44, 77]}')
        for i, j in zip(ii, jj):
#             print(f'dist_map[i, j] = {dist_map[i, j]}')
#             if j < 0:
#                 print(j, end=' ')
#                 j = img.shape[1] - j
#             new_brush_width = int(np.ceil((2 * (1 / max(dist_map[i, j], 1)) * h) / 2))
#             new_brush = ~morphology.disk(new_brush_width).astype(bool)
#             new_h, new_w = new_brush.shape
#             img[i:i + new_h, j:j + new_w] &= new_brush
            img[i:i + h, j:j + w] &= brush
#             mod_h = int(1 / (0.2 + h))
#             mod_w = int(1 / (0.2 + w))
#             mod_brush = ~morphology.disk(max(mod_h, mod_w)).astype(bool)
#             img[i:i + mod_h, j:j + mod_w] &= brush


class OpenFracture(Perturbation):
    """Add fractures to a digit.

    Fractures are added at random locations along the skeleton, while avoiding stroke tips and
    forks, and are locally perpendicular to the pen stroke.
    """

    def __init__(self, thickness: float = 8., prune: float = 7, num_frac: int = 1, angle_to_skeleton: float = np.pi / 2., centres=None, angle_window: int = 2, frac_extension: float = 8., erosion_dilation_amount: float = 0.3, erosion_dilation_window: int = 11, n_erosions_dilations: int = 1., relative=True, debug=False):
        """
        Parameters
        ----------
        thickness : float, optional
            Thickness of the fractures, in low-resolution pixel scale.
        prune : float, optional
            Radius to avoid around stroke tips and forks, in low-resolution pixel scale.
        num_frac : int, optional
            Number of fractures to add.
        """
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac
        self.angle_to_skeleton = angle_to_skeleton
        self.loc_sampler = skeleton.LocationSampler(prune, prune)
        self.angle_window = angle_window
        self.frac_extension = frac_extension
        self.centres = centres
        self.erosion_dilation_amount = erosion_dilation_amount
        self.erosion_dilation_window = erosion_dilation_window
        self.n_erosions_dilations = n_erosions_dilations
        self.relative = relative
        self.debug = debug
    
    def sample_location(self, morph: ImageMorphology):
        """ Uses the LocationSampler on the image's skeleton to get centres.
        """
        try:
            centres = self.loc_sampler.sample(morph, self.num_frac)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, self.num_frac)
        return centres
    
    def set_location(self, centres):
        """ Set the centres attribute of the location.
        """
        self.centres = centres

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        if self.relative:
            y_coords, x_coords = np.where(morph.skeleton > 0)
            y_min, y_max = y_coords.min(), y_coords.max()
            img_h = y_max - y_min

            thickness = (self.thickness / 100.) * morph.mean_thickness
            erosion_dilation_window = int((self.erosion_dilation_window / 100.) * img_h)
            frac_extension = (self.frac_extension / 100.) * img_h
        else:
            thickness = self.thickness
            erosion_dilation_window = self.erosion_dilation_window
            frac_extension = self.frac_extension
        
        original_mean_thickness = morph.mean_thickness
        up_thickness = thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        # create a larger, padded img so that we don't get out of bounds errors when drawing
        frac_img = np.pad(morph.binary_image, pad_width=r, mode='constant', constant_values=False)
        if self.centres is None:
            centres = self.sample_location(morph)
        else:
            centres = self.centres
        for centre in centres:
            p0, p1 = self._endpoints(morph, centre, frac_extension)
            x, y = centre

            radius = int((self.erosion_dilation_amount / 100) * morph.scale * original_mean_thickness)
            mask = np.zeros_like(frac_img).astype(np.bool)
#             p0, p1 = np.sort(np.array([p0, p1]))
#             mask[p0[0]-erosion_dilation_window-brush.shape[0]:p1[0]+erosion_dilation_window+brush.shape[0], p0[1]-erosion_dilation_window-brush.shape[1]:p1[1]+erosion_dilation_window+brush.shape[1]] = True
            mask[centre[0]-erosion_dilation_window-brush.shape[0]:centre[0]+erosion_dilation_window+brush.shape[0], centre[1]-erosion_dilation_window-brush.shape[1]:centre[1]+erosion_dilation_window+brush.shape[1]] = True

            eroded_frac_img = binary_erosion(frac_img, structure=morphology.disk(radius), mask=mask)
            
            if self.debug:
                ax = plt.subplot(1, 6, 1)
                ax.imshow(frac_img)
                ax = plt.subplot(1, 6, 2)
                ax.imshow(mask)
                ax = plt.subplot(1, 6, 3)
                ax.imshow(eroded_frac_img | mask)
                ax = plt.subplot(1, 6, 4)
                ax.imshow(eroded_frac_img)
            
            self._draw_line(eroded_frac_img, p0, p1, brush)
            
            if self.debug:
                ax = plt.subplot(1, 6, 5)
                ax.imshow(eroded_frac_img)
            
            frac_img = binary_dilation(eroded_frac_img, structure=morphology.disk(radius), mask=mask)
            
            def no_grid_no_axis():
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
    
            if self.debug:
                ax = plt.subplot(1, 6, 6)
                ax.imshow(frac_img)
                plt.show()
                
                from skimage.util import invert
                f = plt.figure()
                ax = plt.subplot(1,2,1)
                no_grid_no_axis()
                ax.imshow(invert(morph.binary_image), cmap="gray")
                ax = plt.subplot(1,2,2)
                no_grid_no_axis()
                ax.imshow(invert(frac_img), cmap="gray")
                plt.show()
                plt.close()
                # import pdb; pdb.set_trace()
            
        # return unpadded img by taking off r from all sides
        frac_img = frac_img[r:-r, r:-r]
        return frac_img, (x, y)

    def _endpoints(self, morph, centre, frac_extension):
        angle = skeleton.get_angle(morph.skeleton, *centre, self.angle_window * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + frac_extension * morph.scale
        angle += self.angle_to_skeleton  # Perpendicular to the skeleton by default (pi / 2.)
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush
    
    
class PairOpenFracture(Perturbation):
    """Add fractures to a digit.

    Fractures are added at random locations along the skeleton, while avoiding stroke tips and
    forks, and are locally perpendicular to the pen stroke.
    """

    def __init__(self, thickness: float = 8., prune: float = 7, num_frac: int = 1, angle_to_skeleton: float = np.pi / 2., centres=None, angle_window: int = 2, frac_extension: float = 8., erosion_dilation_amount: float = 0.3, erosion_dilation_window: int = 11, n_erosions_dilations: int = 1., relative=True, debug=False, global_skeleton=None):
        """
        Parameters
        ----------
        thickness : float, optional
            Thickness of the fractures, in low-resolution pixel scale.
        prune : float, optional
            Radius to avoid around stroke tips and forks, in low-resolution pixel scale.
        num_frac : int, optional
            Number of fractures to add.
        """
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac
        self.angle_to_skeleton = angle_to_skeleton
        self.loc_sampler = skeleton.LocationSampler(prune, prune)
        self.angle_window = angle_window
        self.frac_extension = frac_extension
        self.centres = centres
        self.erosion_dilation_amount = erosion_dilation_amount
        self.erosion_dilation_window = erosion_dilation_window
        self.n_erosions_dilations = n_erosions_dilations
        self.relative = relative
        self.debug = debug
        self.global_skeleton = global_skeleton
    
    def sample_location(self, morph: ImageMorphology, sample_size=1):
        """
            Sample a set of possible mid points for the fracture location
            `sample_size`: Number of mid points
        """
        try:
            centres = self.loc_sampler.sample(morph, sample_size)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, sample_size)
        return centres
    
    def set_location(self, centres):
        """ Set the centres attribute of the location.
        """
        self.centres = centres

    def __call__(self, morph1: ImageMorphology, morph2: ImageMorphology, merged_morph: ImageMorphology, global_skeleton) -> (np.ndarray, np.ndarray):
        morph = merged_morph
        if self.relative:
            y_coords, x_coords = np.where(morph.skeleton > 0)
            y_min, y_max = y_coords.min(), y_coords.max()
            img_h = y_max - y_min

            thickness = (self.thickness / 100.) * morph.mean_thickness
            erosion_dilation_window = int((self.erosion_dilation_window / 100.) * img_h)
            frac_extension = (self.frac_extension / 100.) * img_h
        else:
            thickness = self.thickness
            erosion_dilation_window = self.erosion_dilation_window
            frac_extension = self.frac_extension
        
        original_mean_thickness = morph.mean_thickness
        up_thickness = thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        # create a larger, padded img so that we don't get out of bounds errors when drawing
        frac_img1 = np.pad(morph1.binary_image, pad_width=r, mode='constant', constant_values=False)
        frac_img2 = np.pad(morph2.binary_image, pad_width=r, mode='constant', constant_values=False)
        
        if self.centres is None:
            centres = self.sample_location(morph, sample_size=100)
        else:
            centres = self.centres
            
        np.random.shuffle(centres)
        fractured_pair = None
            
        for centre in centres:
            try:
                morph1_p0, morph1_p1 = self._endpoints(morph1, centre, frac_extension)
                morph2_p0, morph2_p1 = self._endpoints(morph2, centre, frac_extension)

                radius = int((self.erosion_dilation_amount / 100) * morph.scale * original_mean_thickness)
                # create mask for local erosion/dilation around sampled centerpoint
                # mask should be the same for both morph1 and morph2 (and hence, frac_img1 and frac_img2)
                mask = np.zeros_like(frac_img1).astype(np.bool)
                mask[centre[0]-erosion_dilation_window-brush.shape[0]:centre[0]+erosion_dilation_window+brush.shape[0], centre[1]-erosion_dilation_window-brush.shape[1]:centre[1]+erosion_dilation_window+brush.shape[1]] = True

                eroded_frac_img1 = binary_erosion(frac_img1, structure=morphology.disk(radius), mask=mask)
                eroded_frac_img2 = binary_erosion(frac_img2, structure=morphology.disk(radius), mask=mask)

                if self.debug:
                    ax1 = plt.subplot(1, 6, 1)
                    ax1.imshow(frac_img1)
                    ax1 = plt.subplot(1, 6, 2)
                    ax1.imshow(mask)
                    ax1 = plt.subplot(1, 6, 3)
                    ax1.imshow(eroded_frac_img1 | mask)
                    ax1 = plt.subplot(1, 6, 4)
                    ax1.imshow(eroded_frac_img1)
                    plt.show()
                    
                    ax2 = plt.subplot(1, 6, 1)
                    ax2.imshow(frac_img2)
                    ax2 = plt.subplot(1, 6, 2)
                    ax2.imshow(mask)
                    ax2 = plt.subplot(1, 6, 3)
                    ax2.imshow(eroded_frac_img2 | mask)
                    ax2 = plt.subplot(1, 6, 4)
                    ax2.imshow(eroded_frac_img2)
                    plt.show()

                self._draw_line(eroded_frac_img1, morph1_p0, morph1_p1, brush)
                self._draw_line(eroded_frac_img2, morph2_p0, morph2_p1, brush)

                if self.debug:
                    ax1 = plt.subplot(1, 6, 5)
                    ax1.imshow(eroded_frac_img1)
                    plt.show()
                    
                    ax2 = plt.subplot(1, 6, 5)
                    ax2.imshow(eroded_frac_img2)
                    plt.show()

                frac_img1 = binary_dilation(eroded_frac_img1, structure=morphology.disk(radius), mask=mask)
                frac_img2 = binary_dilation(eroded_frac_img2, structure=morphology.disk(radius), mask=mask)

                fractured_pair = (frac_img1, frac_img2)
                
                # get nearest neighbor on global skeleton to keep track of the fracture locations
                x_nn, y_nn = skeleton_nearest_neighbor((centre[0], centre[1]), global_skeleton, skel_pixel=True, debug=self.debug)
                                
                def no_grid_no_axis():
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)

                if self.debug:
                    ax1 = plt.subplot(1, 6, 6)
                    ax1.imshow(frac_img1)
                    plt.show()

                    from skimage.util import invert
                    f1 = plt.figure()
                    ax1 = plt.subplot(1,2,1)
                    no_grid_no_axis()
                    ax1.imshow(invert(morph1.binary_image), cmap="gray")
                    ax1 = plt.subplot(1,2,2)
                    no_grid_no_axis()
                    ax1.imshow(invert(frac_img1), cmap="gray")
                    plt.show()

                    
                    ax2 = plt.subplot(1, 6, 6)
                    ax2.imshow(frac_img2)
                    plt.show()
                    plt.close()


                    from skimage.util import invert
                    f2 = plt.figure()
                    ax2 = plt.subplot(1,2,1)
                    no_grid_no_axis()
                    ax2.imshow(invert(morph2.binary_image), cmap="gray")
                    ax2 = plt.subplot(1,2,2)
                    no_grid_no_axis()
                    ax2.imshow(invert(frac_img2), cmap="gray")
                    plt.show()
                    plt.close()
                 
                break
            
            # TODO: make Exceptions to ignore more specific
            except Exception as excpt:
                print(f"Could not generate a good fracture for both images on midpoint {centre}")
                continue
            
        if fractured_pair is None:
            raise PairPerturbException("Unable to fracture the pair of images")
            
        # return unpadded img by taking off r from all sides
        return fractured_pair[0][r:-r, r:-r], fractured_pair[1][r:-r, r:-r], (x_nn, y_nn)

    def _endpoints(self, morph, centre, frac_extension):
        angle = skeleton.get_angle(morph.skeleton, *centre, self.angle_window * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + frac_extension * morph.scale
        angle += self.angle_to_skeleton  # Perpendicular to the skeleton by default (pi / 2.)
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush
    
    
class Bend(Perturbation):
    """Bend a character"""

    def __init__(self, distance: float = 14., dist_shift: float = 10.,  
                 prune_tips=None, prune_forks=None, skeletonizer=None,
                 max_retries = 1,
                 relative=True,
                 interpolation_bending = True,
                 angle_seed=None,
                 midpoints=None,
                 debug=False):
        """
        Parameters
        ----------
        distance : float, optional
            The absolute or relative size of the (random) section of the character that is bent. Absolute or relative is determined by the `relative` argument
        dist_shift: float, optional
            Determines the absolute or relative depth of the bend. Absolute or relative is determined by the `relative` argument
        prune_tips: float, optional
            Radius to avoid around skeleton tips, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        prune_forks: float, optional
            Radius to avoid around skeleton forks, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        max_retries: int, optional
            Retries the bend this many times when any of the two excpetions occur
        relative: bool, optional
            Makes the `distance` and `dist_shift` relative to the image skeleton height
        Angle of the bend is randomly picked
        
        
        """
        self.distance = distance
        self.dist_shift = dist_shift
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks
        self.skeletonizer = skeletonizer
        self.max_retries = max_retries
        self.midpoints = midpoints
        self.relative = relative
        self.angle_seed = angle_seed
        self.interpolation_bending = interpolation_bending
        self.debug = debug
        

    def set_location(self, midpoints):
        """ 
            Set the `midpoints` attribute for the bend location
        """
        self.midpoints = midpoints
    
    def sample_location(self, morph: ImageMorphology, sample_size=1):
        """
            Sample a set of possible mid points for the bend on the skeleton
            `sample_size`: Number of mid points
        """
        sampler = LocationSampler(prune_tips=self.prune_tips, prune_forks=self.prune_forks)
        sampled_midpoints = sampler.sample(morph, num=sample_size)
        # print(sampled_midpoints)
        return sampled_midpoints

    def bend(self, morph: ImageMorphology) -> np.ndarray:
        """
            Bends a given image, specified by `morph`
        """
        if self.midpoints is None:
            sampled_midpoints = self.sample_location(morph, sample_size=100)
        else:
            sampled_midpoints = self.midpoints
        
        # print(sampled_midpoints)
        bent_image = character_bender.bender(morph.binary_image, distance=self.distance, dist_shift=self.dist_shift, 
                      mid_points=sampled_midpoints, skeletonizer=self.skeletonizer, relative=self.relative,
                            new_bend = self.interpolation_bending,
                                   angle_seed = self.angle_seed,
                            debug=self.debug)
        return bent_image
    
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        """
            Bends a given image, specified by `morph`
            Retries the bending once for `BadSkeletonBendException` or  `ChooseSkeletonEndpointsException` Exceptions
        """
        return self.bend(morph)
    

class MergeImagePair:
    """
        Merges a pair of binary images
    """
    def __init__(self, merge_method):
        self.merge_method = merge_method
        assert merge_method in {"union", "intersection"}, f"`merge_method` should be belong to {'union', 'intersection'}"
    
    def __call__(self, morph1, morph2):
        """
            Merges pair of binary images - via union or intersection
        """
        inv_bin_img1 = morph1.binary_image
        inv_bin_img2 = morph2.binary_image
        bin_img1 = character_bender.invert(inv_bin_img1)
        bin_img2 = character_bender.invert(inv_bin_img2)
        merged_bin_img = character_bender.merge_images_np(bin_img1, bin_img2, method=self.merge_method, foreground_color="black")
        
        inv_merged_bin_img = character_bender.invert(merged_bin_img)
        merged_img_morph = ImageMorphology(inv_merged_bin_img)
        return merged_img_morph


class SingleBend(Perturbation):
    """Bend a single base image"""

    def __init__(self, distance: float = 14., dist_shift: float = 10.,  
                 prune_tips=None, prune_forks=None, skeletonizer=None,
                 max_retries=1,
                 relative=True,
                 interpolation_bending=True,
                 angle_seed=None,
                 char=None,
                 midpoints=None,
                 debug=False):
        """
        Parameters
        ----------
        distance : float, optional
            The absolute or relative size of the (random) section of the character that is bent. Absolute or relative is determined by the `relative` argument
        dist_shift: float, optional
            Determines the absolute or relative depth of the bend. Absolute or relative is determined by the `relative` argument
        prune_tips: float, optional
            Radius to avoid around skeleton tips, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        prune_forks: float, optional
            Radius to avoid around skeleton forks, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        max_retries: int, optional
            Retries the bend this many times when exceptions occur
        relative: bool, optional
            Makes the `distance` and `dist_shift` relative to the image skeleton height
        char: Needed for `evaluate_bend_goodness`
        """
        
        self.distance = distance
        self.dist_shift = dist_shift
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks
        self.skeletonizer = skeletonizer
        self.max_retries = max_retries
        self.midpoints = midpoints
        self.relative = relative
        assert char in {'D', 'F', 'G', 'M'}, "`char` argument should be one of {D, F, G, M}"
        self.char = char
        
        self.interpolation_bending = interpolation_bending
        self.debug = debug
        
        if angle_seed is None:
            angle_seed = np.random.randint(1000)
        self.angle_seed = angle_seed  # should be set for the same direction of bend

    def set_location(self, midpoints):
        """Set the `midpoints` attribute for the bend location"""
        self.midpoints = midpoints
    
    def sample_location(self, morph: ImageMorphology, sample_size=1):
        """Sample a set of possible midpoints for the bend on the skeleton"""
        sampler = LocationSampler(prune_tips=self.prune_tips, prune_forks=self.prune_forks)
        sampled_midpoints = sampler.sample(morph, num=sample_size)
        return sampled_midpoints

    def evaluate_bend_goodness(self, approx_area):
        """Eliminate bends that overlap with the curvature of the letter"""
        if self.char == 'G':
            threshold = 35.  # G, manually inspected
        elif self.char == 'M':
            threshold = 16.  # M, manually inspected
        elif self.char == 'F':
            threshold = 20.  # TODO - adjust this as necessary
        else:
            threshold = 16.
        is_good = (approx_area >= threshold)
        return is_good


    def bend(self, morph: ImageMorphology) -> np.ndarray:
        """
            Bends a given image, specified by `morph`
        """
        if self.midpoints is None:
            sampled_midpoints = self.sample_location(morph, sample_size=100)
        else:
            sampled_midpoints = self.midpoints
        
        if self.debug:
            plt.imshow(morph.binary_image, cmap="gray")
            for x,y in sampled_midpoints:
                plt.plot(y, x, "*r")
            plt.title("image with sampled mid-points")
            plt.show()
        
        np.random.shuffle(sampled_midpoints)
        N = len(sampled_midpoints)
        bent_image = None
        N_good_bends = 3
        good_bends = []
        for idx in range(N):
            midpoint = sampled_midpoints[idx:idx+1]
            # print(f"Trying to bend at midpoint {midpoint}")
            # import ipdb; ipdb.set_trace()
            try:
                result = character_bender.bender(
                    morph.binary_image, distance=self.distance, dist_shift=self.dist_shift, 
                    mid_points=midpoint, skeletonizer=self.skeletonizer, relative=self.relative,
                    new_bend=self.interpolation_bending, angle_seed=self.angle_seed, debug=self.debug
                )
                bent_image_attempt = result["binary_image"]
                approx_area = result["approximate_area"]

                is_good_bend = self.evaluate_bend_goodness(approx_area)
                if is_good_bend:
                    # import ipdb; ipdb.set_trace()
                    bent_image = bent_image_attempt
                    break
            except (character_bender.ChooseSkeletonEndpointsException, character_bender.BadSkeletonBendException):
                continue

        if bent_image is None:
            raise SinglePerturbException("Unable to bend the image")

        x_nn, y_nn = midpoint[0]
        return bent_image, (x_nn, y_nn)

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        """Bends a given image, specified by `morph`, with retry logic"""
        assert self.angle_seed is not None, "`angle_bend` should be set for consistent bending"
        return self.bend(morph)

    
    
class PairBend(Perturbation):
    """Bend a pair of base images in the same manner"""

    def __init__(self, distance: float = 14., dist_shift: float = 10.,  
                 prune_tips=None, prune_forks=None, skeletonizer=None,
                 max_retries = 1,
                 relative=True,
                 interpolation_bending = True,
                 angle_seed=None,
                 char=None,
                 midpoints=None,
                 debug=False):
        """
        Parameters
        ----------
        distance : float, optional
            The absolute or relative size of the (random) section of the character that is bent. Absolute or relative is determined by the `relative` argument
        dist_shift: float, optional
            Determines the absolute or relative depth of the bend. Absolute or relative is determined by the `relative` argument
        prune_tips: float, optional
            Radius to avoid around skeleton tips, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        prune_forks: float, optional
            Radius to avoid around skeleton forks, in low-resolution pixel scale (for `skeleton.LocationSampler`)
        max_retries: int, optional
            Retries the bend this many times when any of the two excpetions occur
        relative: bool, optional
            Makes the `distance` and `dist_shift` relative to the image skeleton height
        Angle of the bend is randomly picked
        char: Needed for `evaluate_bend_goodness`
        
        """
        
        # same skeletonizer for both merge and pair images?
        # result in inverted?
        # move the merging outside
        
        self.distance = distance
        self.dist_shift = dist_shift
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks
        self.skeletonizer = skeletonizer
        self.max_retries = max_retries
        self.midpoints = midpoints
        self.relative = relative
        assert char in {'D', 'F', 'G', 'M'}, "`char` argument should be one of {D, F, G, M}"
        self.char = char
        
        self.interpolation_bending = interpolation_bending
        self.debug = debug
        
        if angle_seed is None:
            angle_seed = np.random.randint(1000)
        self.angle_seed = angle_seed # should be set for same direction of bend
        

    def set_location(self, midpoints):
        """ 
            Set the `midpoints` attribute for the bend location
        """
        self.midpoints = midpoints
    
    
    def sample_location(self, morph: ImageMorphology, sample_size=1):
        """
            Sample a set of possible mid points for the bend on the skeleton
            `sample_size`: Number of mid points
        """
        sampler = LocationSampler(prune_tips=self.prune_tips, prune_forks=self.prune_forks)
        sampled_midpoints = sampler.sample(morph, num=sample_size)
        return sampled_midpoints

#     def pick_best_pair(self, bend_pairs):
#         # TODO 
# #         dists = dict()
# #         for i,pair in enumerate(bend_pairs):
# #             (bent_image1, bend1_L2), (bent_image2, bend2_L2) = pair
# #             ds = [bend1_L2, bend2_L2]
# #             if max(ds) >= 2 * min(ds): # heuristic for setting where one of the bends is bad
# #                 continue
# #             dists[sum(ds)] = (bent_image1, bent_image2)
# #         return dists[max(dists)]
#         (bent_image1, bend1_L2), (bent_image2, bend2_L2) = pair
#         return (bent_image1, bent_image2)

    def evaluate_bend_goodness(self, approx_area1, approx_area2):
        """
            To eliminate bends that overlap with the curvature of the letter
        """
        if self.char == 'G':
            threshold = 35. # G  manually inspection of images
        elif self.char == 'M':
            threshold = 16. # M  manually inspection of images
        elif self.char == 'F':
            threshold = 20. # TODO - change this
        is_good = (approx_area1 >= threshold and approx_area2 >= threshold)
        return is_good
            
            
    def bend(self, morph1: ImageMorphology, morph2: ImageMorphology, merged_img_morph: ImageMorphology, global_skeleton) -> (np.ndarray, np.ndarray):
        """
            Bends a given image, specified by `morph`
        """
        if self.midpoints is None:
            sampled_midpoints = self.sample_location(merged_img_morph, sample_size=100)
        else:
            sampled_midpoints = self.midpoints
        
        if self.debug:
            plt.imshow(merged_img_morph.binary_image, cmap="gray")
            for x,y in sampled_midpoints:
                plt.plot(y,x, "*r")
            plt.title("merged image with sampled mid-points")
            plt.show()
        
        np.random.shuffle(sampled_midpoints)
        N = len(sampled_midpoints)
        bent_pair = None
        N_good_bends = 3
        good_bends = []
        for idx in range(N):
            midpoint = sampled_midpoints[idx:idx+1]
            try:
                result1 = character_bender.bender(morph1.binary_image, distance=self.distance, dist_shift=self.dist_shift, 
                      mid_points= midpoint, skeletonizer=self.skeletonizer, relative=self.relative,
                            new_bend = self.interpolation_bending,angle_seed = self.angle_seed,debug=self.debug)
                bent_image1 = result1["binary_image"]
                approx_area_image1 = result1["approximate_area"]
                result2 = character_bender.bender(morph2.binary_image, distance=self.distance, dist_shift=self.dist_shift, 
                      mid_points= midpoint, skeletonizer=self.skeletonizer, relative=self.relative,
                            new_bend = self.interpolation_bending,angle_seed = self.angle_seed,debug=self.debug)
                bent_image2 = result2["binary_image"]
                approx_area_image2 = result2["approximate_area"]
                
#                 print(f"Bend L2: {bend1_L2: .3f}, {bend2_L2: .3f}")
#                 good_bends.append([(bent_image1, bend1_L2), (bent_image2, bend2_L2)])
#                 if len(good_bends) == N_good_bends:
#                     bent_pair = self.pick_best_pair(good_bends)
#                     break
                # print(f"Approximate areas: {approx_area_image1}, {approx_area_image2}\n")
                is_good_bend = self.evaluate_bend_goodness(approx_area_image1, approx_area_image2)
                if is_good_bend:
                    # print("Approx areas: ", approx_area_image1, approx_area_image2)
                    bent_pair = (bent_image1, bent_image2)
                    (x_nn, y_nn) = skeleton_nearest_neighbor(midpoint[0], global_skeleton, skel_pixel=True, debug=self.debug)
                    break
            except (character_bender.ChooseSkeletonEndpointsException, character_bender.BadSkeletonBendException) as excpt:
                # print(f"Could not generate a good bend for both images on midpoint {midpoint}")
                continue
            
        if bent_pair is None:
            raise PairPerturbException("Unable to bend the pair of images")
        return bent_image1, bent_image2, (x_nn, y_nn)
    
    def __call__(self, morph1: ImageMorphology, morph2: ImageMorphology, merged_morph: ImageMorphology, global_skeleton) -> (np.ndarray, np.ndarray):
        """
            Bends a given image, specified by `morph`
            Retries the bending once for `BadSkeletonBendException` or  `ChooseSkeletonEndpointsException` Exceptions
        """
        assert self.angle_seed is not None, "`angle_bend` argument should be set for same direction of bend"
        return self.bend(morph1, morph2, merged_morph, global_skeleton)

    
