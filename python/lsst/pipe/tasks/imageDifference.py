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

import numpy as np

import lsst.afw.image
import lsst.afw.math
import lsst.afw.table
import lsst.ip.diffim
import lsst.obs.base
import lsst.pex.config
import lsst.pipe.base
from lsst.pipe.base import connectionTypes

from .measurePsf import MeasurePsfTask

__all__ = ["AlardLuptonSubtractConfig", "AlardLuptonSubtractTask"]


class AlardLuptonSubtractConnections(lsst.pipe.base.PipelineTaskConnections,
                                     dimensions=("instrument", "visit", "detector"),
                                     defaultTemplates={"coaddName": "deep",
                                                       "fakesType": ""}):
    science = connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    sources = connectionTypes.Input(
        doc="Sources measured on the science exposure; "
            "used to select sources for making the matching kernel.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}src"
    )
    template = connectionTypes.Input(
        doc="Input template exposure to subtract.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp"
    )

    scoreExposure = connectionTypes.Output(
        doc="Output AL likelihood or Zogy score image",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_scoreExp",
    )

    difference = connectionTypes.Output(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    matchedTemplate = connectionTypes.Output(
        doc="Warped template used to create `subtractedExposure`.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_matchedExp",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doWriteScoreExp:
            self.outputs.remove("scoreExposure")


class AlardLuptonSubtractConfig(lsst.pipe.base.PipelineTaskConfig,
                                pipelineConnections=AlardLuptonSubtractConnections):
    makeKernel = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.MakeKernelTask,
        doc="Task to construct a matching kernel for convolution.",
    )
    requiredTemplateFraction = lsst.pex.config.Field(
        dtype=float, default=0.1,
        doc="Do not attempt to run task if template covers less than this fraction of pixels."
        "Setting to 0 will always attempt image subtraction"
    )
    doDecorrelation = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Perform diffim decorrelation to undo pixel correlation due to A&L "
        "kernel convolution (AL only)? If True, also update the diffim PSF."
    )
    decorrelate = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.DecorrelateALKernelTask,
        doc="Task to decorrelate the image difference.",
    )

    doWriteScoreExp = lsst.pex.config.Field(
        dtype=bool,
        default=False,
        doc="Write AL likelihood or Zogy score exposure?"
    )

    measureTemplatePsf = lsst.pex.config.Field(
        dtype=bool,
        default=False,
        doc="Measure the PSF of the template before constructing the matching kernel."
    )
    measurePsf = lsst.pex.config.ConfigurableField(
        target=MeasurePsfTask,
        doc="Measure PSF",
    )
    forceConvolveTemplate = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Always convolve the template, even if the science image has better seeing."
    )

    def setDefaults(self):
        # defaults are OK for catalog and diacatalog

        self.makeKernel.kernel.name = "AL"
        self.makeKernel.kernel.active.fitForBackground = True
        self.makeKernel.kernel.active.spatialKernelOrder = 1
        self.makeKernel.kernel.active.spatialBgOrder = 2


class AlardLuptonSubtractTask(lsst.pipe.base.PipelineTask):
    """Subtract a template from a science image using the Alard & Lupton (1998)
    algorithm.
    """
    ConfigClass = AlardLuptonSubtractConfig
    _DefaultName = "alardLuptonSubtract"

    def __init__(self, butler=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("decorrelate")
        self.makeSubtask("makeKernel")

        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.makeSubtask("measurePsf", schema=self.schema)

    @lsst.utils.inheritDoc(lsst.pipe.base.PipelineTask)
    def runQuantum(self, butlerQC: lsst.pipe.base.ButlerQuantumContext,
                   inputRefs: lsst.pipe.base.InputQuantizedConnection,
                   outputRefs: lsst.pipe.base.OutputQuantizedConnection):
        inputs = butlerQC.get(inputRefs)
        self.log.info("Processing %s", butlerQC.quantum.dataId)
        self.checkTemplateIsSufficient(inputs['template'])

        outputs = self.run(template=inputs['template'],
                           science=inputs['science'],
                           sources=inputs['sources'])
        butlerQC.put(outputs, outputRefs)

    def run(self, template, science, sources):

        convolutionControl = lsst.afw.math.ConvolutionControl()
        convolutionControl.setDoNormalize(False)

        # This could become a PipelineTask that produces kernelSources as output
        kernelSources = self.makeKernel.selectKernelSources(template, science,
                                                            candidateList=sources,
                                                            preconvolved=False)

        if self.config.measureTemplatePsf:
            templatePsfSize0 = self._getFwhmPix(template)
            self.log.info("Template PSF size prior to measurement: %f", templatePsfSize0)
            exposureIdInfo = lsst.obs.base.ExposureIdInfo()
            self.measurePsf.run(exposure=template, sources=sources, matches=None,
                                expId=exposureIdInfo.expId)

        sciencePsfSize = self._getFwhmPix(science)
        templatePsfSize = self._getFwhmPix(template)
        self.log.info("Science PSF size: %f", sciencePsfSize)
        self.log.info("Template PSF size: %f", templatePsfSize)
        if self.config.forceConvolveTemplate:
            self.log.info("forceConvolveTemplate=True is set, convolving template image.")
            subtractRes = self.convolveTemplateSubtract(template, science, kernelSources, convolutionControl)
        else:
            if sciencePsfSize < templatePsfSize:
                self.log.info("Template PSF size is the greater, convolving science image.")
                subtractRes = self.convolveScienceSubtract(template, science,
                                                           kernelSources, convolutionControl)
            else:
                self.log.info("Science PSF size is the greater, convolving template image.")
                subtractRes = self.convolveTemplateSubtract(template, science,
                                                            kernelSources, convolutionControl)
        if self.config.doWriteScoreExp:
            preconvolvedScience, preconvolveKernel = self.preconvolveScience(science, convolutionControl)
            scoreExposure = self.preconvolveSubtract(template, preconvolvedScience, preconvolveKernel,
                                                     kernelSources, convolutionControl)
            subtractRes.scoreExposure = scoreExposure
        return subtractRes

    def convolveScienceSubtract(self, template, science, kernelSources, convolutionControl):
        """Subtract a template from a science image using the Alard & Lupton (1998)
        algorithm (convolving the science image with the template PSF).
        """
        kernel = self.makeKernel.run(science, template, kernelSources).psfMatchingKernel

        # Science and template bboxes should be identical!
        convolvedScience = lsst.afw.image.MaskedImageF(science.getBBox())
        lsst.afw.math.convolve(convolvedScience, science.maskedImage, kernel, convolutionControl)

        difference = lsst.afw.image.ExposureF(convolvedScience, science.getWcs())
        difference.maskedImage -= template.maskedImage
        difference.setPsf(template.psf)
        difference.setWcs(science.getWcs())
        difference.setFilterLabel(science.getFilterLabel())

        matchedScience = self._makeExposure(convolvedScience, science)
        if self.config.doDecorrelation:
            self.log.info("Decorrelating image difference.")
            difference = self.decorrelate.run(matchedScience, template, difference, kernel,
                                              templateMatched=False,
                                              preConvMode=False,
                                              preConvKernel=None,
                                              spatiallyVarying=False).correctedExposure
        else:
            self.log.info("NOT decorrelating image difference.")
        return lsst.pipe.base.Struct(difference=difference,
                                     matchedTemplate=template,
                                     matchedScience=matchedScience,
                                     )

    def convolveTemplateSubtract(self, template, science, kernelSources, convolutionControl):
        """Subtract a template from a science image using the Alard & Lupton (1998)
        algorithm (convolving the template image with the science PSF).
        """
        kernel = self.makeKernel.run(template, science, kernelSources).psfMatchingKernel

        # Science and template bboxes should be identical!
        convolvedTemplate = lsst.afw.image.MaskedImageF(science.getBBox())
        lsst.afw.math.convolve(convolvedTemplate, template.maskedImage, kernel, convolutionControl)

        difference = lsst.afw.image.ExposureF(science, deep=True)
        difference.maskedImage -= convolvedTemplate
        difference.setWcs(science.getWcs())
        difference.setFilterLabel(science.getFilterLabel())

        matchedTemplate = self._makeExposure(convolvedTemplate, science)
        if self.config.doDecorrelation:
            # NOTE: This takes the unconvolved template!
            self.log.info("Decorrelating image difference.")
            difference = self.decorrelate.run(science, template, difference, kernel,
                                              templateMatched=True,
                                              preConvMode=False,
                                              preConvKernel=None,
                                              spatiallyVarying=False).correctedExposure
        else:
            self.log.info("NOT decorrelating image difference.")
        return lsst.pipe.base.Struct(difference=difference,
                                     matchedTemplate=matchedTemplate,
                                     matchedScience=science,
                                     )

    def preconvolveSubtract(self, template, preconvolvedScience, preconvolveKernel,
                            kernelSources, convolutionControl):
        """Subtract a template from a science image using the Alard & Lupton (1998)
        algorithm (convolving the template image with the science PSF).
        """
        kernel = self.makeKernel.run(template, preconvolvedScience, kernelSources,
                                     preconvolved=True).psfMatchingKernel

        # Science and template bboxes should be identical!
        convolvedTemplate = lsst.afw.image.MaskedImageF(preconvolvedScience.getBBox())
        lsst.afw.math.convolve(convolvedTemplate, template.maskedImage, kernel, convolutionControl)

        scoreExposure = lsst.afw.image.ExposureF(preconvolvedScience, deep=True)
        scoreExposure.maskedImage -= convolvedTemplate
        scoreExposure.setWcs(preconvolvedScience.getWcs())
        scoreExposure.setFilterLabel(preconvolvedScience.getFilterLabel())

        if self.config.doDecorrelation:
            # NOTE: This takes the unconvolved template!
            self.log.info("Decorrelating Score exposure.")
            scoreExposure = self.decorrelate.run(preconvolvedScience, template, scoreExposure, kernel,
                                                 templateMatched=True,
                                                 preConvMode=True,
                                                 preConvKernel=preconvolveKernel,
                                                 spatiallyVarying=False).correctedExposure
        else:
            self.log.info("NOT decorrelating Score exposure.")
        return scoreExposure

    def preconvolveScience(self, science, convolutionControl):
        # cannot convolve in place, so need a new image anyway
        convolvedScience = lsst.afw.image.MaskedImageF(science.getBBox())
        psfAvgPos = science.psf.getAveragePosition()
        preconvolveKernel = science.psf.getLocalKernel(psfAvgPos)
        # convolve with science exposure's PSF model
        lsst.afw.math.convolve(convolvedScience, science.maskedImage,
                               preconvolveKernel,
                               convolutionControl)
        badPix = np.isnan(convolvedScience.image.array)
        if badPix is not None:
            noData = lsst.afw.image.Mask.getPlaneBitMask("NO_DATA")
            convolvedScience.image.array[badPix] = 0
            convolvedScience.mask.array[badPix] |= noData

        return self._makeExposure(convolvedScience, science), preconvolveKernel

    @staticmethod
    def _getFwhmPix(exposure):
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        psf = exposure.getPsf()
        psfAvgPos = psf.getAveragePosition()
        psfSize = psf.computeShape(psfAvgPos).getDeterminantRadius()*sigma2fwhm
        return psfSize

    @staticmethod
    def _makeExposure(maskedImage, exposure):
        newExposure = lsst.afw.image.ExposureF(maskedImage, exposure.getWcs())
        newExposure.setPsf(exposure.psf)
        newExposure.setFilterLabel(exposure.getFilterLabel())
        newExposure.setPhotoCalib(exposure.getPhotoCalib())
        return newExposure

    def checkTemplateIsSufficient(self, templateExposure):
        """Raise NoWorkFound if template coverage < requiredTemplateFraction

        Parameters
        ----------
        templateExposure : `lsst.afw.image.ExposureF`
            The template exposure to check

        Raises
        ------
        NoWorkFound
            Raised if fraction of good pixels, defined as not having NO_DATA
            set, is less then the configured requiredTemplateFraction
        """
        # Count the number of pixels with the NO_DATA mask bit set
        # counting NaN pixels is insufficient because pixels without data are often intepolated over)
        pixNoData = np.count_nonzero(templateExposure.mask.array
                                     & templateExposure.mask.getPlaneBitMask('NO_DATA'))
        pixGood = templateExposure.getBBox().getArea() - pixNoData
        self.log.info("template has %d good pixels (%.1f%%)", pixGood,
                      100*pixGood/templateExposure.getBBox().getArea())

        if pixGood/templateExposure.getBBox().getArea() < self.config.requiredTemplateFraction:
            message = ("Insufficient Template Coverage. (%.1f%% < %.1f%%) Not attempting subtraction. "
                       "To force subtraction, set config requiredTemplateFraction=0." % (
                           100*pixGood/templateExposure.getBBox().getArea(),
                           100*self.config.requiredTemplateFraction))
            raise lsst.pipe.base.NoWorkFound(message)
