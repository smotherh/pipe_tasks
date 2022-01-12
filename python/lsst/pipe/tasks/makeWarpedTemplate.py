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

import lsst.afw.geom
from lsst.meas.algorithms import WarpedPsf
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import GetCoaddAsTemplateTask
from lsst.skymap import BaseSkyMap
import lsst.utils

__all__ = ["MakeWarpedTemplateConfig", "MakeWarpedTemplateTask"]


class MakeWarpedTemplateTaskConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("instrument", "visit", "detector", "skymap"),
                                        defaultTemplates={"coaddName": "deep",
                                                          "warpTypeSuffix": "",
                                                          "fakesType": ""}):

    exposure = pipeBase.connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )

    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for template exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap", ),
        storageClass="SkyMap",
    )
    coaddExposures = pipeBase.connectionTypes.Input(
        doc="Input template to match and subtract from the exposure",
        dimensions=("tract", "patch", "skymap", "band"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Coadd{warpTypeSuffix}",
        multiple=True,
        deferLoad=True
    )
    dcrCoadds = pipeBase.connectionTypes.Input(
        doc="Input DCR template to match and subtract from the exposure",
        name="{fakesType}dcrCoadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "subfilter"),
        multiple=True,
        deferLoad=True
    )
    outputExposure = pipeBase.connectionTypes.Output(
        doc="Warped template used to create `subtractedExposure`.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp{warpTypeSuffix}",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if self.config.useDcrCoadds:
            self.inputs.remove("coaddExposures")
        else:
            self.inputs.remove("dcrCoadds")


class MakeWarpedTemplateConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=MakeWarpedTemplateTaskConnections):
    """Config for MakeWarpedTemplateTask.
    """
    useDcrCoadds = pexConfig.Field(
        doc="Use DCR coadds to make the template.",
        default=False,
        dtype=bool
    )
    getTemplate = pexConfig.ConfigurableField(
        target=GetCoaddAsTemplateTask,
        doc="Subtask to retrieve template exposure and sources",
    )
    warpingConfig = pexConfig.ConfigField("Config for warping exposures to a common alignment",
                                          lsst.afw.math.WarperConfig)

    def validate(self):
        pexConfig.Config.validate(self)
        if hasattr(self.getTemplate, "coaddName"):
            if (self.useDcrCoadds and self.getTemplate.coaddName != "dcr") or \
                    (not self.useDcrCoadds and self.getTemplate.coaddName == "dcr"):
                raise ValueError("Mis-matched `getTemplate.coaddName` and `useDcrCoadds` values in config.")


class MakeWarpedTemplateTask(pipeBase.PipelineTask):
    """Warp and assemble coadds to match the WCS and bounding box
    of a science exposure.
    """
    ConfigClass = MakeWarpedTemplateConfig
    _DefaultName = "makeWarpedTemplate"

    def __init__(self, butler=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("getTemplate")
        self._warper = lsst.afw.math.Warper.fromConfig(self.config.warpingConfig)

    @lsst.utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC: pipeBase.ButlerQuantumContext,
                   inputRefs: pipeBase.InputQuantizedConnection,
                   outputRefs: pipeBase.OutputQuantizedConnection):
        inputs = butlerQC.get(inputRefs)
        self.log.info("Processing %s", butlerQC.quantum.dataId)
        if self.config.useDcrCoadds:
            templateExposures = inputRefs.dcrCoadds
        else:
            templateExposures = inputRefs.coaddExposures

        scienceExposure = inputs['exposure']
        templateExposure = self.getTemplate.runQuantum(
            scienceExposure, butlerQC, inputRefs.skyMap, templateExposures).exposure
        # Warp PSF before overwriting exposure
        templatePsf = templateExposure.getPsf()
        xyTransform = lsst.afw.geom.makeWcsPairTransform(templateExposure.getWcs(),
                                                         scienceExposure.getWcs())
        psfWarped = WarpedPsf(templatePsf, xyTransform)
        templateExposure = self._warper.warpExposure(scienceExposure.getWcs(),
                                                     templateExposure,
                                                     destBBox=scienceExposure.getBBox())
        templateExposure.setPsf(psfWarped)
        output = pipeBase.Struct(outputExposure=templateExposure)

        butlerQC.put(output, outputRefs)
