#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import unittest

import lsst.utils.tests
import lsst.geom as geom
import lsst.afw.geom as afwGeom
from lsst.pipe.tasks.selectImages import WcsSelectImagesTask, SelectStruct
from lsst.pipe.tasks.coaddBase import CoaddBaseTask


class KeyValue:

    """Mixin to provide __getitem__ of key/value pair"""

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __getitem__(self, key):
        if key != self._key:
            raise KeyError("Unrecognised key in %s: %s vs %s" % (self.__class__.__name__, key, self._key))
        return self._value


class DummyPatch:

    """Quacks like a lsst.skymap.PatchInfo"""

    def __init__(self, xy0, dims):
        self._outerBBox = geom.Box2I(xy0, dims)

    def getOuterBBox(self):
        return self._outerBBox


class DummyTract(KeyValue):

    """Quacks like a lsst.skymap.TractInfo"""

    def __init__(self, patchId, patch, wcs):
        super(DummyTract, self).__init__(patchId, patch)
        self._wcs = wcs

    def getPatchInfo(self, patchId):
        return self[patchId]

    def getWcs(self):
        return self._wcs


class DummySkyMap(KeyValue):

    """Quacks like a lsst.skymap.BaseSkyMap"""

    def __init__(self, tractId, tract):
        super(DummySkyMap, self).__init__(tractId, tract)


class DummyDataRef:

    """Quacks like a lsst.daf.persistence.ButlerDataRef"""

    def __init__(self, dataId, **data):
        self.dataId = dataId
        self._data = data

    def get(self, dataType):
        return self._data[dataType]


# Common defaults for createPatch and createImage
CENTER = geom.SpherePoint(0, 90, geom.degrees)
ROTATEAXIS = geom.SpherePoint(0, 0, geom.degrees)
DIMS = geom.Extent2I(3600, 3600)
SCALE = 0.5*geom.arcseconds


def createPatch(
    tractId=1, patchId=(2, 3),  # Tract and patch identifier, for dataId
    dims=DIMS,                # Patch dimensions (Extent2I)
    xy0=geom.Point2I(1234, 5678),  # Patch xy0 (Point2I)
    center=CENTER,                  # ICRS sky position of center (lsst.afw.geom.SpherePoint)
    scale=SCALE                     # Pixel scale (Angle)
):
    crpix = geom.Point2D(xy0) + geom.Extent2D(dims)*0.5
    cdMatrix = afwGeom.makeCdMatrix(scale=scale)
    wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=center, cdMatrix=cdMatrix)
    patch = DummyPatch(xy0, dims)
    tract = DummyTract(patchId, patch, wcs)
    skymap = DummySkyMap(tractId, tract)
    dataRef = DummyDataRef({'tract': tractId, 'patch': ",".join(map(str, patchId))}, deepCoadd_skyMap=skymap)
    return dataRef


def createImage(
    dataId={"name": "foobar"},          # Data identifier
    center=CENTER,                      # ICRS sky position of center (lsst.afw.geom.SpherePoint)
    rotateAxis=ROTATEAXIS,              # Rotation axis (lsst.afw.geom.SpherePoint)
    rotateAngle=0*geom.degrees,         # Rotation angle/distance to move (Angle)
    dims=DIMS,                          # Image dimensions (Extent2I)
    scale=SCALE,                        # Pixel scale (Angle)
    noWcs=False,                        # Set WCS to None
):
    crpix = geom.Point2D(geom.Extent2D(dims)*0.5)
    center = center.rotated(rotateAxis, rotateAngle)
    cdMatrix = afwGeom.makeCdMatrix(scale=scale)
    if noWcs:
        wcs = None
    else:
        wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=center, cdMatrix=cdMatrix)
    return SelectStruct(DummyDataRef(dataId), wcs, geom.Box2I(geom.Point2I(0, 0),
                                                              geom.Extent2I(dims[0], dims[1])))


class WcsSelectImagesTestCase(unittest.TestCase):

    def check(self, patchRef, selectData, doesOverlap, numExpected=1):
        config = CoaddBaseTask.ConfigClass()
        config.select.retarget(WcsSelectImagesTask)
        task = CoaddBaseTask(config=config, name="CoaddBase")
        dataRefList = task.selectExposures(patchRef, selectDataList=[selectData])
        numExpected = numExpected if doesOverlap else 0
        self.assertEqual(len(dataRefList), numExpected)

    def testIdentical(self):
        self.check(createPatch(), createImage(), True)

    def testImageContains(self):
        self.check(createPatch(), createImage(scale=2*SCALE), True)

    def testImageContained(self):
        self.check(createPatch(), createImage(scale=0.5*SCALE), True)

    def testDisjoint(self):
        self.check(createPatch(),
                   createImage(center=geom.SpherePoint(0, -90, geom.degrees)),
                   False)

    def testIntersect(self):
        self.check(createPatch(), createImage(rotateAngle=0.5*geom.Extent2D(DIMS).computeNorm()*SCALE),
                   True)

    def testWcsNone(self):
        self.check(createPatch(), createImage(noWcs=True), True, numExpected=0)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
