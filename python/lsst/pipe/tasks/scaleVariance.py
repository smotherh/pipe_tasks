#
# LSST Data Management System
# Copyright 2018 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from contextlib import contextmanager
import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.meas.algorithms import SubtractBackgroundTask

__all__ = ["ScaleVarianceConfig", "ScaleVarianceTask"]


class ScaleVarianceConfig(Config):
    background = ConfigurableField(target=SubtractBackgroundTask, doc="Background subtraction")
    maskPlanes = ListField(
        dtype=str,
        default=["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"],
        doc="Mask planes for pixels to ignore when scaling variance",
    )
    limit = Field(dtype=float, default=10.0, doc="Maximum variance scaling value to permit")

    def setDefaults(self):
        self.background.binSize = 32
        self.background.useApprox = False
        self.background.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.background.ignoredPixelMask = ["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"]


class ScaleVarianceTask(Task):
    """Scale the variance in a MaskedImage

    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This Task
    attempts to correct for this by scaling the variance plane to match
    the observed variance in the image. This is not perfect (because we're
    not tracking the covariance) but it's simple and is often good enough.

    The task implements a pixel-based and an image-based correction estimator.
    """
    ConfigClass = ScaleVarianceConfig
    _DefaultName = "scaleVariance"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.makeSubtask("background")

    @contextmanager
    def subtractedBackground(self, maskedImage, refMaskedImage=None):
        """Context manager for subtracting the background

        We need to subtract the background so that the entire image
        (apart from objects, which should be clipped) will have the
        image/sqrt(variance) distributed about zero.

        This context manager subtracts the background, and ensures it
        is restored on exit.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image+mask+variance to have background subtracted and restored.
        refMaskedImage : `lsst.afw.image.MaskedImage`, optional
            Image from which to determine which pixels to mask.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        context : context manager
            Context manager that ensure the background is restored.
        """
        bg = self.background.fitBackground(maskedImage)
        bgImage = bg.getImageF(self.background.config.algorithm, self.background.config.undersampleStyle)
        maskedImage -= bgImage
        if refMaskedImage:
            bgRef = self.background.fitBackground(refMaskedImage)
            bgImageRef = bgRef.getImageF(self.background.config.algorithm,
                                         self.background.config.undersampleStyle)
            refMaskedImage -= bgImageRef
        try:
            yield
        finally:
            maskedImage += bgImage
            if refMaskedImage:
                refMaskedImage += bgImageRef

    def run(self, maskedImage, refMaskedImage=None):
        """Rescale the variance in a maskedImage in place.

        Parameters
        ----------
        maskedImage :  `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor. The image
            is modified in place.
        refMaskedImage : `lsst.afw.image.MaskedImage`
            Image to determine which pixels to mask when determining variance.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        factor : `float`
            Variance rescaling factor.

        Raises
        ------
        RuntimeError
            If the estimated variance rescaling factor by both methods exceed the
            configured limit.

        Notes
        -----
        The task calculates and applies the pixel-based correction unless
        it is over the ``config.limit`` threshold. In this case, the image-based
        method is applied.
        """
        if refMaskedImage is None:
            refMaskedImage = maskedImage

        with self.subtractedBackground(maskedImage, refMaskedImage):
            factor = self.pixelBased(maskedImage, refMaskedImage)
            if factor > self.config.limit:
                self.log.warning("Pixel-based variance rescaling factor (%f) exceeds configured limit (%f); "
                                 "trying image-based method", factor, self.config.limit)
                factor = self.imageBased(maskedImage, refMaskedImage)
                if factor > self.config.limit:
                    raise RuntimeError("Variance rescaling factor (%f) exceeds configured limit (%f)" %
                                       (factor, self.config.limit))
            self.log.info("Renormalizing variance by %f", factor)
            maskedImage.variance *= factor
        return factor

    def computeScaleFactors(self, maskedImage, refMaskedImage=None):
        """Calculate and return both variance scaling factors without modifying the image.

        Parameters
        ----------
        maskedImage :  `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.
        refMaskedImage : `lsst.afw.image.MaskedImage`, optional
            Image from which to determine which pixels to mask.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        R : `lsst.pipe.base.Struct`
          - ``pixelFactor`` : `float` The pixel based variance rescaling factor
            or 1 if all pixels are masked or invalid.
          - ``imageFactor`` : `float` The image based variance rescaling factor
            or 1 if all pixels are masked or invalid.
        """
        with self.subtractedBackground(maskedImage, refMaskedImage):
            pixelFactor = self.pixelBased(maskedImage, refMaskedImage)
            imageFactor = self.imageBased(maskedImage, refMaskedImage)
        return Struct(pixelFactor=pixelFactor, imageFactor=imageFactor)

    def pixelBased(self, maskedImage, refMaskedImage=None):
        """Determine the variance rescaling factor from pixel statistics

        We calculate SNR = image/sqrt(variance), and the distribution
        for most of the background-subtracted image should have a standard
        deviation of unity. We use the interquartile range as a robust estimator
        of the SNR standard deviation. The variance rescaling factor is the
        factor that brings that distribution to have unit standard deviation.

        This may not work well if the image has a lot of structure in it, as
        the assumptions are violated. In that case, use an alternate
        method.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.
        refMaskedImage : `lsst.afw.image.MaskedImage`, optional
            Image from which to determine which pixels to mask.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        factor : `float`
            Variance rescaling factor or 1 if all pixels are masked or non-finite.

        """
        if refMaskedImage is None:
            refMaskedImage = maskedImage
        maskVal = refMaskedImage.mask.getPlaneBitMask(self.config.maskPlanes)
        isGood = (((refMaskedImage.mask.array & maskVal) == 0)
                  & np.isfinite(refMaskedImage.image.array)
                  & np.isfinite(refMaskedImage.variance.array)
                  & (refMaskedImage.variance.array > 0))

        nGood = np.sum(isGood)
        self.log.debug("Number of selected background pixels: %d of %d.", nGood, isGood.size)
        if nGood < 2:
            # Not enough good data, np.percentile needs at least 2 points
            # to estimate a range
            return 1.0
        # Robust measurement of stdev using inter-quartile range
        snr = maskedImage.image.array[isGood]/np.sqrt(maskedImage.variance.array[isGood])
        q1, q3 = np.percentile(snr, (25, 75))
        stdev = 0.74*(q3 - q1)
        return stdev**2

    def imageBased(self, maskedImage, refMaskedImage=None):
        """Determine the variance rescaling factor from image statistics

        We calculate average(SNR) = stdev(image)/median(variance), and
        the value should be unity. We use the interquartile range as a robust
        estimator of the stdev. The variance rescaling factor is the
        factor that brings this value to unity.

        This may not work well if the pixels from which we measure the
        standard deviation of the image are not effectively the same pixels
        from which we measure the median of the variance. In that case, use
        an alternate method.

        Parameters
        ----------
        maskedImage :  `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.
        refMaskedImage : `lsst.afw.image.MaskedImage`, optional
            Image from which to determine which pixels to mask.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        factor : `float`
            Variance rescaling factor or 1 if all pixels are masked or non-finite.
        """
        if refMaskedImage is None:
            refMaskedImage = maskedImage

        maskVal = refMaskedImage.mask.getPlaneBitMask(self.config.maskPlanes)
        isGood = (((refMaskedImage.mask.array & maskVal) == 0)
                  & np.isfinite(refMaskedImage.image.array)
                  & np.isfinite(refMaskedImage.variance.array)
                  & (refMaskedImage.variance.array > 0))
        nGood = np.sum(isGood)
        self.log.debug("Number of selected background pixels: %d of %d.", nGood, isGood.size)
        if nGood < 2:
            # Not enough good data, np.percentile needs at least 2 points
            # to estimate a range
            return 1.0
        # Robust measurement of stdev
        q1, q3 = np.percentile(maskedImage.image.array[isGood], (25, 75))
        ratio = 0.74*(q3 - q1)/np.sqrt(np.median(maskedImage.variance.array[isGood]))
        return ratio**2
