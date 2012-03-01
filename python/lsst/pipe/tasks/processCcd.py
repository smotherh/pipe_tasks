#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Instrumental Signature Removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass,
                                      doc="Calibration (inc. high-threshold detection and measurement)")
    detection = pexConfig.ConfigField(dtype=measAlg.SourceDetectionTask.ConfigClass,
                                      doc="Low-threshold detection for final measurement")
    measurement = pexConfig.ConfigField(dtype=measAlg.SourceMeasurementTask.ConfigClass,
                                        doc="Final source measurement on low-threshold detections")

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")

    def setDefaults(self):
        self.doWriteIsr = False
        self.isr.methodList = ["doConversionForIsr", "doSaturationDetection",
                               "doOverscanCorrection", "doVariance", "doFlatCorrection"]
        self.isr.doWrite = False

class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("isr", IsrTask)
        self.makeSubtask("calibrate", CalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                             schema=self.schema, algMetadata=self.algMetadata)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
            rawExposure = sensorRef.get("raw")
            isrRes = self.isr.run(rawExposure, calibSet)
            self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
            exposure = self.isr.doCcdAssembly([isrRes.postIsrExposure])
            self.display("ccdAssembly", exposure=exposure)
            if self.config.doWriteIsr:
                sensorRef.put(exposure, 'postISRCCD')
        else:
            exposure = None

        if self.config.doCalibrate:
            if exposure is None:
                exposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(exposure)
            exposure = calib.exposure
            if self.config.doWriteCalibrate:
<<<<<<< HEAD
                sensorRef.put(exposure, 'calexp')
                # FIXME: SourceCatalog not butlerized
                #sensorRef.put(calib.sources, 'icSrc')
=======
                sensorRef.put(afwDet.PersistableSourceVector(calib.sources), 'icSrc')
>>>>>>> master
                if calib.psf is not None:
                    sensorRef.put(calib.psf, 'psf')
                if calib.apCorr is not None:
                    # FIXME: ApertureCorrection not butlerized
                    #sensorRef.put(calib.apCorr, 'apcorr')
                    pass
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    # FIXME: BaseCatalog (i.e. normalized match vector) not butlerized
                    #sensorRef.put(normalizedMatches, 'icMatch')
        else:
            calib = None

        if self.config.doDetection:
            if exposure is None:
                exposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                exposure.setPsf(sensorRef.get('psf'))
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, exposure)
        else:
            sources = None

        if self.config.doMeasurement:
            assert(sources)
            assert(exposure)
            if calib is None:
                apCorr = None # FIXME: should load from butler
                if self.measurement.doApplyApCorr:
                    self.log.log(self.log.WARN, "Cannot load aperture correction; will not be applied.")
            else:
                apCorr = calib.apCorr
            self.measurement.run(exposure, sources, apCorr)

        if self.config.doWriteSources:
            # FIXME: SourceCatalog not butlerized
            #sensorRef.put(phot.sources, 'src')
            pass

        if self.config.doWriteCalibrate:
            sensorRef.put(ccdExposure, 'calexp')
            
        return pipeBase.Struct(
            exposure = exposure,
            calib = calib,
            sources = sources,
        )
