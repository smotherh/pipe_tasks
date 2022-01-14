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

import itertools
import numpy as np
import unittest

from lsst.afw import image as afwImage
from lsst.pipe.tasks.scaleVariance import ScaleVarianceTask
from lsst.pipe.tasks.compute_noise_correlation import (ComputeNoiseCorrelationTask,
                                                       ComputeNoiseCorrelationConfig)
import lsst.utils.tests


class NoiseVarianceTestCase(lsst.utils.tests.TestCase):

    def _generateImage(self, rho1, rho2):
        # Create a correlated noise field using simple translations
        np.random.seed(12345)
        noise = np.random.randn(512, 512).astype(np.float32)
        # Solve for the kernel parameters and generate correlated noise
        r2 = rho1**2 + rho2**2
        if r2 > 0:
            k = 0.5*(1 + np.sqrt(1 - 4*r2))/r2
            a1, a2 = k*rho1, k*rho2
            corr_noise = noise + a1*np.roll(noise, 1, axis=0) + a2*np.roll(noise, 1, axis=1)
        else:
            a1, a2 = 0, 0
            corr_noise = noise
        image = afwImage.ImageF(array=corr_noise[1:-1, 1:-1])
        variance = afwImage.ImageF(510, 510, (1+a1**2+a2**2))
        mi = afwImage.MaskedImageF(image=image, variance=variance)
        return mi

    @lsst.utils.tests.methodParameters(rho=((0., 0.), (-0.2, 0.0), (0.0, 0.1), (0.15, 0.25), (0.25, -0.15)))
    def testScaleVariance(self, rho):
        task = ScaleVarianceTask()
        rho1, rho2 = rho
        mi = self._generateImage(rho1, rho2)
        scaleFactors = task.computeScaleFactors(mi)
        self.assertFloatsAlmostEqual(scaleFactors.pixelFactor, scaleFactors.imageFactor, atol=1e-6)
        self.assertFloatsAlmostEqual(scaleFactors.pixelFactor, 1.0, rtol=2e-2)

    @lsst.utils.tests.methodParameters(rho=((0., 0.), (0.2, 0.0), (0.0, -0.1), (0.15, 0.25), (-0.25, 0.15)))
    def testComputeCovariance(self, rho):
        config = ComputeNoiseCorrelationConfig()
        config.size = 5
        task = ComputeNoiseCorrelationTask(config=config)

        rho1, rho2 = rho
        mi = self._generateImage(rho1, rho2)
        cov_matrix = task.run(mi)

        err = np.std([cov_matrix[i, j] for i, j in itertools.product(range(5), range(5)) if (i + j > 1)])

        self.assertLess(abs(cov_matrix[1, 0]/cov_matrix[0, 0] - rho1), 2*err)
        self.assertLess(abs(cov_matrix[0, 1]/cov_matrix[0, 0] - rho2), 2*err)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
