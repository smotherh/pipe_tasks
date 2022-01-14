# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from contextlib import contextmanager
import itertools
import numpy as np

import lsst.afw.image
from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task
from lsst.meas.algorithms import SubtractBackgroundTask

__all__ = ["ComputeNoiseCorrelationConfig", "ComputeNoiseCorrelationTask"]


class ComputeNoiseCorrelationConfig(Config):
    background = ConfigurableField(target=SubtractBackgroundTask, doc="Background subtraction")
    maskPlanes = ListField(
        dtype=str,
        default=["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"],
        doc="Mask planes for pixels to ignore when scaling variance",
    )
    size = Field(
        dtype=int,
        default=5,
        doc="Size of the covariance matrix to produce"
    )
    scaleEmpiricalVariance = Field(
        dtype=bool,
        default=False,
        doc="Scale down the covariance matrix by the empirical variance of the background?"
    )
    subtractEmpiricalMean = Field(
        dtype=bool,
        default=False,
        doc="Subtract the empirical mean in addition to the background?"
    )

    def setDefaults(self):
        self.background.binSize = 32
        self.background.useApprox = False
        self.background.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.background.ignoredPixelMask = ["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"]


class ComputeNoiseCorrelationTask(Task):
    """Compute the noise correlation coefficients in a MaskedImage

    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This Task computes
    a matrix of correlation coefficients of a desired size.
    """
    ConfigClass = ComputeNoiseCorrelationConfig
    _DefaultName = "computeNoiseCorrelation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("background")
        self.background: SubtractBackgroundTask  # type hint

    @contextmanager
    def subtractedBackground(self, maskedImage: lsst.afw.image.MaskedImage):
        """Context manager for subtracting the background

        We need to subtract the background so that the entire image
        (apart from objects, which should be clipped) will have the
        image/sqrt(variance) distributed about zero with unit variance.

        This context manager subtracts the background, and ensures it
        is restored on exit.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image+mask+variance to have background subtracted and restored.

        Returns
        -------
        context : context manager
            Context manager that ensure the background is restored.
        """
        bg = self.background.fitBackground(maskedImage)
        bgImage = bg.getImageF(self.background.config.algorithm, self.background.config.undersampleStyle)
        maskedImage -= bgImage
        try:
            yield
        finally:
            maskedImage += bgImage

    def run(self, maskedImage: lsst.afw.image.MaskedImage,
            refMaskedImage: lsst.afw.image.MaskedImage = None) -> np.ndarray:
        """Compute the correlation matrix from a maskedImage.

        Parameters
        ----------
        maskedImage :  `~lsst.afw.image.MaskedImage`
            Image for which to determine the correlation matrix.
        refMaskedImage : `~lsst.afw.image.MaskedImage`, optional
            Image from which to determine which pixels to mask.
            If None, it defaults to ``maskedImage``.

        Returns
        -------
        corr_matrix : `numpy.ndarray`
            Correlation matrix of the maskedImage.

        Raises
        ------
        RuntimeError
            Raised if ``refMaskedImage`` is provided and does not have the same
            dimensions as ``maskedImage``.
        """
        corr_matrix = np.empty((self.config.size+1, self.config.size+1), dtype=np.float32)
        with self.subtractedBackground(maskedImage):
            if refMaskedImage is None:
                refMaskedImage = maskedImage
            elif refMaskedImage.getDimensions() != maskedImage.getDimensions():
                raise RuntimeError("Reference image has different dimensions than input image")

        corr_matrix = self._pixelBased(maskedImage, refMaskedImage)
        return corr_matrix

    def _pixelBased(self,
                    maskedImage: lsst.afw.image.MaskedImage,
                    refMaskedImage: lsst.afw.image.MaskedImage) -> np.ndarray:
        """Determine correlation coefficients between pixels

           This is the concrete routine that does the computation.

        Parameters
        ----------
        maskedImage : `~lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.
        refMaskedImage : `~lsst.afw.image.MaskedImage`
            Image from which to determine which pixels to mask.

        Returns
        -------
        corr_matrix : `numpy.ndarray`
            Correlation matrix of the maskedImage.
        """
        maskVal = refMaskedImage.mask.getPlaneBitMask(self.config.maskPlanes)
        isGood = (((refMaskedImage.mask.array & maskVal) == 0)
                  & np.isfinite(refMaskedImage.image.array)
                  & np.isfinite(refMaskedImage.variance.array)
                  & (refMaskedImage.variance.array > 0))

        nGood = np.sum(isGood)
        self.log.debug("Number of selected background pixels: %d of %d.", nGood, isGood.size)

        normalized_arr = maskedImage.image.array/np.sqrt(maskedImage.variance.array)
        normalized_arr[~isGood] = np.nan

        corr_matrix = np.empty((self.config.size+1, self.config.size+1), dtype=np.float32)

        for dx, dy in itertools.product(range(self.config.size+1), range(self.config.size+1)):
            sliceX = slice(None, -dx) if dx != 0 else slice(None, None)
            sliceY = slice(None, -dy) if dy != 0 else slice(None, None)
            arr1 = normalized_arr[sliceX, sliceY]

            sliceX = slice(dx, None) if dx != 0 else slice(None, None)
            sliceY = slice(dy, None) if dy != 0 else slice(None, None)
            arr2 = normalized_arr[sliceX, sliceY]

            if self.config.subtractEmpiricalMean:
                arr1 -= np.nanmean(arr1)
                arr2 -= np.nanmean(arr2)
            if self.config.scaleEmpiricalVariance:
                # Do not use nanstd direct, as it will subtract the
                # empirical mean regardless of config set.
                arr1 /= np.nanmean(arr1**2)**0.5
                arr2 /= np.nanmean(arr2**2)**0.5

            cov = np.nanmean(arr1*arr2)
            # Adjust for the bias in the estimator. Temporary reference:
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Practical_issues
            # TODO: Explain this in the DMTN-215 (DM-33418).
            cov *= 1. + 0.5*(1-cov**2)/(~np.isnan(arr1*arr2)).sum()

            corr_matrix[dx, dy] = cov

        return corr_matrix
