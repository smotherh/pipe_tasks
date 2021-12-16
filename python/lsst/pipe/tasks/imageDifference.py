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

import lsst.afw.image
import lsst.afw.math
import lsst.ip.diffim
import lsst.pex.config
import lsst.pipe.base
from lsst.pipe.base import connectionTypes


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
        name="{fakesType}{coaddName}Diff_warpedExp"
    )

    difference = connectionTypes.Output(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )


class AlardLuptonSubtractConfig(lsst.pex.config.Config,
                                pipelineConnections=AlardLuptonSubtractConnections):
    decorrelate = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.DecorrelateALKernelTask,
        doc="Initial detections used to feed stars to kernel fitting",
    )


class AlardLuptonSubtractTask(lsst.pipe.base.PipelineTask):
    """Subtract a template from a science image using the Alard & Lupton (1998)
    algorithm.
    """
    ConfigClass = AlardLuptonSubtractConfig
    _DefaultName = "alardLuptonSubtract"

    def __init__(self):
        self.makeSubtask("decorrelate")

    def run(self, science, sources, template):
        # TODO: do we really pass in config here???
        # are these things subtasks, or just called directly?
        # should they even be Tasks themselves, or just classes?
        if something:
            AlardLuptonConvolveScienceSubtractTask.run(science, sources, template, config)
        else:
            AlardLuptonConvolveTemplateSubtractTask.run(science, sources, template, config)


class AlardLuptonConvolveTemplateSubtractTask(lsst.pipe.base.Task):
    """Subtract a template from a science image using the Alard & Lupton (1998)
    algorithm (convolving the template image with the science PSF).
    """
    def run(self, science, sources, template):
        kernel = lsst.ip.diffim.makeKernel(science, template, sources)

        # Science and template bboxes should be identical!
        convolved_template = lsst.afw.image.MaskedImageF(science.getBBox())
        convolutionControl = lsst.afw.math.ConvolutionControl()
        convolutionControl.setDoNormalize(False)
        lsst.afw.math.convolve(convolved_template, template.maskedImage, kernel, convolutionControl)

        # TODO: do we need deep=True here?
        diff = lsst.afw.image.ExposureF(science, deep=True)
        diff.maskedImage -= convolved_template.maskedImage

        # NOTE: This takes the unconvolved template!
        decorrelated = self.decorrelate.run(science, template, diff, kernel, templateMatched=True)
        
        return decorrelated.correctedExposure


class AlardLuptonConvolveScienceSubtractTask(lsst.pipe.base.Task):
    """Subtract a template from a science image using the Alard & Lupton (1998)
    algorithm (convolving the science image with the template PSF).
    """
    def run(self, science, sources, template):
        kernel = lsst.ip.diffim.makeKernel(science, template, sources)

        # Science and template bboxes should be identical!
        convolved_science = lsst.afw.image.MaskedImageF(science.getBBox())
        convolutionControl = lsst.afw.math.ConvolutionControl()
        convolutionControl.setDoNormalize(False)
        lsst.afw.math.convolve(convolved_science, science.maskedImage, kernel, convolutionControl)

        # TODO: do we need deep=True here?
        diff = lsst.afw.image.ExposureF(convolved_science, deep=True)
        diff.maskedImage -= template.maskedImage
        diff.setPsf(template.psf)

        decorrelated = self.decorrelate.run(convolved_science, template, diff, kernel, templateMatched=False)
        
        return decorrelated.correctedExposure
