"""
This module extends the core set of 2-D transformations into 3-D transforms
"""

import numpy as np
from numpy import ma
from matplotlib._path import affine_transform
from numpy.linalg import inv

from weakref import WeakKeyDictionary
import warnings
try:
    set
except NameError:
    from sets import Set as set

import cbook
from path import Path
from _path import count_bboxes_overlapping_bbox, update_path_extents

DEBUG = False
if DEBUG:
    import warnings

MaskedArray = ma.MaskedArray

class BboxBase3D(BboxBase):
    """
    This extends the basic BboxBase class to the 3D case.
    This interface, like its origin, is a read-only interface.
    A mutable bounding box is provided by the :class:'Bbox3D' class.

    The canonical representation is as three points, with no
    restrictions on their ordering.  Convenience properties are
    provided to get the left, bottom, front, right, top and back edges
    and width, height and depth, but these are not stored explicity.
    """
    if DEBUG:
        def _check(points):
            if ma.isMaskedArray(points):
                warnings.warn("Bbox bounds are a masked array.")
            points = np.asarray(points)
            if any(points[1,:] - points[0, :] == 0) :
                warnings.warn("Singular Bbox.")
        _check = staticmethod(_check)

    def frozen(self):
        return Bbox3D(self.get_points().copy())
    frozen.__doc__ = TransformNode.__doc__

    def is_unit(self):
        """
        Returns True if the :class:`Bbox` is the unit bounding box
        from (0, 0, 0) to (1, 1, 1).
        """
        return self.get_points().flatten().tolist() == [0., 0., 0., 1., 1., 1.]

    def _get_z0(self):
        return self.get_points()[0, 2]
    z0 = property(_get_z0, None, None, """
         (property) :attr:`z0` is the first of the pair of *z* coordinates that
         define the bounding box.  :attr:`z0` is not guaranteed to be
         less than :attr:`z1`.  If you require that, use :attr:`zmin`.""")

    def _get_z1(self):
        return self.get_points()[1, 2]
    z1 = property(_get_z1, None, None, """
         (property) :attr:`z1` is the second of the pair of *z* coordinates that
         define the bounding box.  :attr:`z1` is not guaranteed to be
         greater than :attr:`z0`.  If you require that, use :attr:`zmax`.""")

    _get_p0.__doc__ = """
         (property) :attr:`p0` is the first pair of (*x*, *y*, *z*)
         coordinates that define the bounding box.  It is not guaranteed
         to be the bottom-left-front corner.  For that, use :attr:`min`."""

    _get_p1.__doc__ = """
         (property) :attr:`p1` is the second pair of (*x*, *y*, *z*)
         coordinates that define the bounding box.  It is not guaranteed
         to be the top-right-back corner.  For that, use :attr:`max`."""

    def _get_zmin(self):
        return min(self.get_points()[:, 2])
    zmin = property(_get_zmin, None, None, """
        (property) :attr:`zmin` is the front edge of the bounding box.""")

    def _get_zmax(self):
        return max(self.get_points()[:, 2])
    zmax = property(_get_zmax, None, None, """
        (property) :attr:`zmax` is the back edge of the bounding box.""")

    def _get_min(self):
        return self.get_points().min(axis=0).tolist()
    min = property(_get_min, None, None, """
        (property) :attr:`min` is the bottom-left-front corner of the bounding
        box.""")

    def _get_max(self):
        return self.get_points().max(axis=0).tolist()
    max = property(_get_max, None, None, """
        (property) :attr:`max` is the top-right-back corner of the bounding box.""")

    def _get_intervalz(self):
        return self.get_points()[:, 2]
    intervalz = property(_get_intervalz, None, None, """
        (property) :attr:`intervalz` is the pair of *z* coordinates that define
        the bounding box. It is not guaranteed to be sorted from back to
        front.""")

    def _get_depth(self):
        points = self.get_points()
        return points[1, 2] - points[0, 2]
    depth = property(_get_depth, None, None, """
        (property) The depth of the bounding box.  It may be negative if
        :attr:`z1` < :attr:`z0`.""")

    _get_size.__doc__ = """
        (property) The width, height and depth of the bounding box.  May be
        negative, in the same way as :attr:`width`, :attr:`height`
        and :attr:`depth`."""

    def _get_bounds(self):
        x0, y0, z0, x1, y1, z1 = self.get_points().flatten()
        return (x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
    bounds = property(_get_bounds, None, None, """
        (property) Returns (:attr:`x0`, :attr:`y0`, :attr:`z0`,
        :attr:`width`, :attr:`height`, :attr:`depth`).""")

    def _get_extents(self):
        return self.get_points().flatten().copy()
    extents = property(_get_extents, None, None, """
        (property) Returns (:attr:`x0`, :attr:`y0`, :attr:`z0`,
        :attr:`x1`, :attr:`y1`, :attr:`z1`).""")

    def containsz(self, z):
        """
        Returns True if *z* is between or equal to :attr:`z0` and
        :attr:`z1`.
        """
        z0, z1 = self.intervalz
        return ((z0 < z1
                 and (z >= z0 and z <= z1))
                or (z >= z1 and z <= z0))

    def contains(self, x, y, z):
        """
        Returns *True* if (*x*, *y*, *z*) is a coordinate inside the
        bounding box or on its edge.
        """
        return Bbox.contains(self, x, y) and self.containsz(z)

    def overlaps(self, other):
        """
        Returns True if this bounding box overlaps with the given
        bounding box *other*.
        """
        ax1, ay1, az1, ax2, ay2, az2 = self._get_extents()
        bx1, by1, bz1, bx2, by2, bz2 = other._get_extents()

        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if az2 < az1:
            az2, az1 = az1, az2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        if bz2 < bz1:
            bz2, bz1 = bz1, bz2

        return not ((bx2 < ax1) or
                    (by2 < ay1) or
                    (bz2 < az1) or
                    (bx1 > ax2) or
                    (by1 > ay2) or
                    (bz1 > az2))

    def fully_containsz(self, z):
        """
        Returns True if *z* is between but not equal to :attr:`z0` and
        :attr:`z1`.
        """
        z0, z1 = self.intervalz
        return ((z0 < z1
                 and (z > z0 and z < z1))
                or (z > z1 and z < z0))

    def fully_contains(self, x, y, z):
        """
        Returns True if (*x*, *y*, *z*) is a coordinate inside the bounding
        box, but not on its edge.
        """
        return Bbox.fully_contains(self, x, y) and self.fully_containsz(z)

    def fully_overlaps(self, other):
        """
        Returns True if this bounding box overlaps with the given
        bounding box *other*, but not on its edge alone.
        """
        ax1, ay1, az1, ax2, ay2, az2 = self._get_extents()
        bx1, by1, bz1, bx2, by2, bz2 = other._get_extents()

        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if az2 < az1:
            az2, az1 = az1, az2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        if bz2 < bz1:
            bz2, bz1 = bz1, bz2

        return not ((bx2 <= ax1) or
                    (by2 <= ay1) or
                    (bz2 <= bz1) or
                    (bx1 >= ax2) or
                    (by1 >= ay2) or
                    (bz1 >= bz2))

    def transformed(self, transform):
        """
        Return a new :class:`Bbox3D` object, statically transformed by
        the given transform.
        """
        return Bbox3D(transform.transform(self.get_points()))

    def inverse_transformed(self, transform):
        """
        Return a new :class:`Bbox3D` object, statically transformed by
        the inverse of the given transform.
        """
        return Bbox3D(transform.inverted().transform(self.get_points()))

    coefs = {'C':  (0.5, 0.5, 0.5),
             'SW': (0,0, 0.5),
             'S':  (0.5, 0, 0.5),
             'SE': (1.0, 0, 0.5),
             'E':  (1.0, 0.5, 0.5),
             'NE': (1.0, 1.0, 0.5),
             'N':  (0.5, 1.0, 0.5),
             'NW': (0, 1.0, 0.5),
             'W':  (0, 0.5, 0.5)}

    def anchored(self, c, container = None):
        """
        Return a copy of the :class:`Bbox3D`, shifted to position *c*
        within a container.

        *c*: may be either:

          * a sequence (*cx*, *cy*, *cz*) where *cx*, *cy* and *cz* range
            from 0 to 1, where 0 is left, bottom or front and 1 is right,
            top or back

          * a string:
            - 'C' for centered
            - 'S' for bottom-center
            - 'SE' for bottom-left
            - 'E' for left
            - etc.

        Optional argument *container* is the box within which the
        :class:`Bbox3D` is positioned; it defaults to the initial
        :class:`Bbox3D`.
        """
        # TODO: Need to expand the list of available anchor points.
        if container is None:
            container = self
        l, b, f, w, h, d = container.bounds
        if isinstance(c, str):
            cx, cy, cz = self.coefs[c]
        else:
            cx, cy, cz = c
        L, B, F, W, H, D = self.bounds
        return Bbox(self._points +
                    [(l + cx * (w-W)) - L,
                     (b + cy * (h-H)) - B,
                     (f + cz * (d-D)) - F])

    def shrunk(self, mx, my, mz=1):
        """
        Return a copy of the :class:`Bbox3D`, shrunk by the factor *mx*
        in the *x* direction and the factor *my* in the *y* direction,
        and the factor *mz* in the *z* direction.
        The lower left front corner of the box remains unchanged.  Normally
        *mx*, *my*, *mz* will be less than 1, but this is not enforced.
        """
        w, h, d = self.size
        return Bbox3D([self._points[0],
                       self._points[0] + [mx * w, my * h, mz * d]])

    def shrunk_to_aspect(self, box_aspect, container = None, fig_aspect = 1.0):
        """
        Return a copy of the :class:`Bbox3D`, shrunk so that it is as
        large as it can be while having the desired aspect ratio,
        *box_aspect*.  If the box coordinates are relative---that
        is, fractions of a larger box such as a figure---then the
        physical aspect ratio of that figure is specified with
        *fig_aspect*, so that *box_aspect* can also be given as a
        ratio of the absolute dimensions, not the relative dimensions.
        """
        # TODO: Figure out a proper way to handle this.
        assert box_aspect > 0 and fig_aspect > 0
        if container is None:
            container = self
        w, h, d = container.size
        H = w * box_aspect/fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect/box_aspect
            H = h
        return Bbox3D([self._points[0],
                       self._points[0] + (W, H, d)])

    def splitx(self, *args):
        """
        e.g., ``bbox.splitx(f1, f2, ...)``

        Returns a list of new :class:`Bbox3D` objects formed by
        splitting the original one with horizontal lines at fractional
        positions *f1*, *f2*, ...
        """
        boxes = []
        xf = [0] + list(args) + [1]
        x0, y0, z0, x1, y1, z1 = self._get_extents()
        w = x1 - x0
        for xf0, xf1 in zip(xf[:-1], xf[1:]):
            boxes.append(Bbox3D([[x0 + xf0 * w, y0, z0], [x0 + xf1 * w, y1, z1]]))
        return boxes

    def splity(self, *args):
        """
        e.g., ``bbox.splitx(f1, f2, ...)``

        Returns a list of new :class:`Bbox3D` objects formed by
        splitting the original one with horizontal lines at fractional
        positions *f1*, *f2*, ...
        """
        boxes = []
        yf = [0] + list(args) + [1]
        x0, y0, z0, x1, y1, z1 = self._get_extents()
        h = y1 - y0
        for yf0, yf1 in zip(yf[:-1], yf[1:]):
            boxes.append(Bbox3D([[x0, y0 + yf0 * h, z0], [x1, y0 + yf1 * h, z1]]))
        return boxes

    def splitz(self, *args):
        """
        e.g., ``bbox.splitz(f1, f2, ...)``

        Returns a list of new :class:`Bbox3D` objects formed by
        splitting the original one with vertical lines at fractional
        positions *f1*, *f2*, ...
        """
        boxes = []
        yf = [0] + list(args) + [1]
        x0, y0, z0, x1, y1, z1 = self._get_extents()
        d = z1 - z0
        for zf0, zf1 in zip(zf[:-1], zf[1:]):
            boxes.append(Bbox3D([[x0, y0, z0 + zf0 * d], [x1, y1, z0 + zf1 * d]]))
        return boxes

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the :class:`Bbox3D`.

        *vertices* is a Nx3 Numpy array.
        """
        if vertices.size == 0:
            return 0
        vertices = np.asarray(vertices)
        x0, y0, z0, x1, y1, z1 = self._get_extents()
        dx0 = np.sign(vertices[:, 0] - x0)
        dy0 = np.sign(vertices[:, 1] - y0)
        dz0 = np.sign(vertices[:, 2] - z0)
        dx1 = np.sign(vertices[:, 0] - x1)
        dy1 = np.sign(vertices[:, 1] - y1)
        dz1 = np.sign(vertices[:, 2] - z1)
        inside = (abs(dx0 + dx1) + abs(dy0 + dy1) + abs(dz0 + dz1)) <= 3
        return np.sum(inside)

    def expanded(self, sw, sh, sd=1):
        """
        Return a new :class:`Bbox3D` which is this :class:`Bbox3D`
        expanded around its center by the given factors *sw*,
        *sh* and *sd*.
        """
        width = self.width
        height = self.height
        depth = self.depth
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        deltad = (sd * depth - depth) / 2.0
        a = np.array([[-deltaw, -deltah, -deltad], [deltaw, deltah, deltad]])
        return Bbox3D(self._points + a)

    def padded(self, p):
        """
        Return a new :class:`Bbox3D` that is padded on all sides by
        the given value.
        """
        points = self.get_points()
        return Bbox3D(points + [[-p, -p, -p], [p, p, p]])

    def translated(self, tx, ty, tz=0):
        """
        Return a copy of the :class:`Bbox3D`, statically translated by
        *tx*, *ty*, and *tz*.
        """
        return Bbox3D(self._points + (tx, ty, tz))

    def corners(self):
        """
        Return an array of points which are the eight corners of this
        box.  For example, if this :class:`Bbox3D` is defined by
        the points (*a*, *b*, *c*) and (*d*, *e*, *f*), :meth:`corners`
        returns (*a*, *b*, *c*), (*a*, *e*, *c*),
                (*a*, *b*, *f*), (*a*, *e*, *f*),
                (*d*, *b*, *c*), (*d*, *b*, *c*),
                (*d*, *b*, *f*), (*d*, *e*, *f*).
        """
        x0, y0, z0, x1, y1, z1 = self.get_points().flatten()
        return np.array([[x0, y0, z0], [x0, y1, z0],
                         [x0, y0, z1], [x0, y1, z1],
                         [x1, y0, z0], [x1, y1, z0],
                         [x1, y0, z1], [x1, y1, z1]])

    def rotated(self, radians, phi=0):
        """
        Return a new bounding box that bounds a rotated version of
        this bounding box by the given radians in azimuth (x & y),
        and by the given radians in phi (z).  The new bounding box
        is still aligned with the axes, of course.
        """
        corners = self.corners()
        corners_rotated = Affine3D().rotate(radians, phi).transform(corners)
        bbox = Bbox3D.unit()
        bbox.update_from_data_xyz(corners_rotated, ignore=True)
        return bbox

    @staticmethod
    def union(bboxes):
        """
        Return a :class:`Bbox3D` that contains all of the given bboxes.
        """
        assert(len(bboxes))

        if len(bboxes) == 1:
            return bboxes[0]

        x0 = np.inf
        y0 = np.inf
        z0 = np.inf
        x1 = -np.inf
        y1 = -np.inf
        z1 = -np.inf

        for bbox in bboxes:
            points = bbox.get_points()
            xs = points[:, 0]
            ys = points[:, 1]
            zs = points[:, 2]
            x0 = min(x0, np.min(xs))
            y0 = min(y0, np.min(ys))
            z0 = min(z0, np.min(zs))
            x1 = max(x1, np.max(xs))
            y1 = max(y1, np.max(ys))
            z1 = max(z1, np.max(zs))

        return Bbox3D.from_extents(x0, y0, z0, x1, y1, z1)


class Bbox3D(BboxBase3D, Bbox):
    """
    A mutable bounding box for 3D space.
    """

    def __init__(self, points):
        """
        *points*: a 2x3 numpy array of the form [[x0, y0, z0], [x1, y1, z1]]

        If you need to create a :class:`Bbox3D` object from another form
        of data, consider the static methods :meth:`unit`,
        :meth:`from_bounds` and :meth:`from_extents`.
        """
        BboxBase3D.__init__(self)
        self._points = np.asarray(points, np.float_)
        self._minpos = np.array([0.0000001, 0.0000001, 0.0000001])
        self._ignore = True
        # it is helpful in some contexts to know if the bbox is a
        # default or has been mutated; we store the orig points to
        # support the mutated methods
        self._points_orig = self._points.copy()
    if DEBUG:
        ___init__ = __init__
        def __init__(self, points):
            self._check(points)
            self.___init__(points)

        def invalidate(self):
            self._check(self._points)
            TransformNode.invalidate(self)

    _unit_values = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float_)
    @staticmethod
    def unit():
        """
        (staticmethod) Create a new unit :class:`Bbox3D` from (0, 0, 0) to
        (1, 1, 1).
        """
        return Bbox3D(Bbox3D._unit_values.copy())

    @staticmethod
    def from_bounds(x0, y0, z0, width, height, depth):
        """
        (staticmethod) Create a new :class:`Bbox3D` from *x0*, *y0*,
        *z0*, *width*, *height*, *depth*

        *width*, *height* and *depth* may be negative.
        """
        return Bbox3D.from_extents(x0, y0, z0, x0 + width, y0 + height, z0 + depth)

    @staticmethod
    def from_extents(*args):
        """
        (staticmethod) Create a new Bbox3D from *left*, *bottom*, *front*,
        *right*, *top* and *back*.

        The *y*-axis increases upwards and *z*-axis increase depth-wise.
        """
        points = np.array(args, dtype=np.float_).reshape(2, 3)
        return Bbox3D(points)

    def __repr__(self):
        return 'Bbox3D(%s)' % repr(self._points)
    __str__ = __repr__

    def ignore(self, value):
        """
        Set whether the existing bounds of the box should be ignored
        by subsequent calls to :meth:`update_from_data` or
        :meth:`update_from_data_xyz`.

        *value*:

           - When True, subsequent calls to :meth:`update_from_data`
             will ignore the existing bounds of the :class:`Bbox3D`.

           - When False, subsequent calls to :meth:`update_from_data`
             will include the existing bounds of the :class:`Bbox3D`.
        """
        self._ignore = value

    def update_from_data(self, x, y, z, ignore=None):
        """
        Update the bounds of the :class:`Bbox3D` based on the passed in
        data.  After updating, the bounds will have positive *width*,
        *height* and *depth*; *x0*, *y0*, and *z0* will be the minimal
        values.

        *x*: a numpy array of *x*-values

        *y*: a numpy array of *y*-values

        *z*: a numpy array of *z*-values

        *ignore*:
           - when True, ignore the existing bounds of the :class:`Bbox3D`.
           - when False, include the existing bounds of the :class:`Bbox3D`.
           - when None, use the last value passed to :meth:`ignore`.
        """
        warnings.warn(
            "update_from_data requires a memory copy -- please replace with update_from_data_xyz")
        xyz = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))))
        return self.update_from_data_xyz(xyz, ignore)

    def update_from_path(self, path, ignore=None,
                         updatex=True, updatey=True):
        """
        Update the bounds of the :class:`Bbox3D` based on the passed in
        data.  After updating, the bounds will have positive *width*,
        *height* and *depth*; *x0*, *y0* and *z0* will be the minimal values.

        *path*: a :class:`~matplotlib.path.Path` instance

        *ignore*:
           - when True, ignore the existing bounds of the :class:`Bbox3D`.
           - when False, include the existing bounds of the :class:`Bbox3D`.
           - when None, use the last value passed to :meth:`ignore`.

        *updatex*: when True, update the x values

        *updatey*: when True, update the y values
        """
        # TODO: Get 3D version...
        if ignore is None:
            ignore = self._ignore

        if path.vertices.size == 0:
            return

        points, minpos, changed = update_path_extents(
            path, None, self._points[:, 0:2], self._minpos, ignore)

        if changed:
            self.invalidate()
            if updatex:
                self._points[:,0] = points[:,0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:,1] = points[:,1]
                self._minpos[1] = minpos[1]


    # TODO: Do we need to define a update_from_data_xyz
    update_from_data_xyz = Bbox.update_from_data_xy

    def _set_z0(self, val):
        self._points[0, 2] = val
        self.invalidate()
    z0 = property(BboxBase3D._get_z0, _set_z0)

    def _set_z1(self, val):
        self._points[1, 2] = val
        self.invalidate()
    z1 = property(BboxBase3D._get_z1, _set_z1)

    def _set_intervalz(self, interval):
        self._points[:, 2] = interval
        self.invalidate()
    intervalz = property(BboxBase3D._get_intervalz, _set_intervalz)

    def _set_bounds(self, bounds):
        x0, y0, z0, w, h, d = bounds
        points = np.array([[x0, y0, z0], [x0+w, y0+h, z0+d]], np.float_)
        if np.any(self._points != points):
            self._points = points
            self.invalidate()
    bounds = property(BboxBase3D._get_bounds, _set_bounds)

    def _get_minposz(self):
        return self._minpos[2]
    minposz = property(_get_minposz)

    Bbox.get_points.__doc__ = """
        Get the points of the bounding box directly as a numpy array
        of the form: [[x0, y0, z0], [x1, y1, z1]].
        """

    Bbox.set_points.__doc__ = """
        Set the points of the bounding box directly from a numpy array
        of the form: [[x0, y0, z0], [x1, y1, z1]].  No error checking is
        performed, as this method is mainly for internal use.
        """

    Bbox.set.__doc__ = """
        Set this bounding box from the "frozen" bounds of another
        :class:`Bbox3D`.
        """

    def mutated(self):
        'return whether the bbox has changed since init'
        return Bbox.mutated(self) or self.mutatedz()

    def mutatedz(self):
        'return whether the z-limits have changed since init'
        return any(self._points[:,2]!=self._points_orig[:,2])


class TransformedBbox3D(BboxBase3D, BboxBase, TransformedBbox):
    """
    A :class:`Bbox3D` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """
    def __init__(self, bbox, transform):
        """
        *bbox*: a child :class:`Bbox3D`

        *transform*: a 3D :class:`Transform`
        """
        assert bbox.is_bbox
        assert isinstance(transform, Transform)
        assert transform.input_dims == 3
        assert transform.output_dims == 3

        BboxBase3D.__init__(self)
        self._bbox = bbox
        self._transform = transform
        self.set_children(bbox, transform)
        self._points = None

    def __repr__(self):
        return "TransformedBbox3D(%s, %s)" % (self._bbox, self._transform)
    __str__ = __repr__


class Affine3DBase(Affine2DBase, AffineBase):
    # TODO: Update this doc!
    """
    The base class of all 3D affine transformations.

    3D affine transformations are performed using a 4x4 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 3D
    affine transformation, use :class:`Affine3D`.

    Subclasses of this class will generally only need to override a
    constructor and :meth:`get_matrix` that generates a custom 4x4 matrix.
    """

    input_dims = 3
    output_dims = 3

    def frozen(self):
        return Affine3D(self.get_matrix().copy())
    frozen.__doc__ = AffineBase.frozen.__doc__

    # TODO: Gotta figure this out...
    def _get_is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0
    is_separable = property(_get_is_separable)


    # TODO: Gotta figure this out...
    def to_values(self):
        """
        Return the values of the matrix as a sequence (a,b,c,d,e,f.....)
        """
        mtx = self.get_matrix()
        return tuple(mtx[:3].swapaxes(0, 1).flatten())

    # TODO: Gotta figure this out...
    @staticmethod
    def matrix_from_values(a, b, c, d, e, f):
        """
        (staticmethod) Create a new transformation matrix as a 3x3
        numpy array of the form::

          a c e
          b d f
          0 0 1
        """
        return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], np.float_)

    def inverted(self):
        if self._inverted is None or self._invalid:
            mtx = self.get_matrix()
            self._inverted = Affine3D(inv(mtx))
            self._invalid = 0
        return self._inverted
    inverted.__doc__ = AffineBase.inverted.__doc__


class Affine3D(Affine3DBase, Affine2DBase):
    """
    A mutable 3D affine transformation.
    """

    def __init__(self, matrix = None):
        # TODO: Update doc!
        """
        Initialize an Affine transform from a 4x4 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        Affine3DBase.__init__(self)
        if matrix is None:
            matrix = np.identity(3)
        elif DEBUG:
            matrix = np.asarray(matrix, np.float_)
            assert matrix.shape == (4, 4)
        self._mtx = matrix
        self._invalid = 0

    def __repr__(self):
        return "Affine3D(%s)" % repr(self._mtx)
    __str__ = __repr__

    def __cmp__(self, other):
        if (isinstance(other, Affine3D) and
            (self.get_matrix() == other.get_matrix()).all()):
            return 0
        return -1

    @staticmethod
    # TODO: Fix this docs!
    def from_values(a, b, c, d, e, f):
        """
        (staticmethod) Create a new Affine3D instance from the given
        values::

          a c e
          b d f
          0 0 1
        """
        return Affine3D(
            np.array([a, c, e, b, d, f, 0.0, 0.0, 1.0], np.float_)
            .reshape((4,4)))

    def get_matrix(self):
        # TODO: Update docs!
        """
        Get the underlying transformation matrix as a 4x4 numpy array::

          a c e
          b d f
          0 0 1
        """
        self._invalid = 0
        return self._mtx

    def set_matrix(self, mtx):
        # TODO: Update docs!
        """
        Set the underlying transformation matrix from a 4x4 numpy array::

          a c e
          b d f
          0 0 1
        """
        self._mtx = mtx
        self.invalidate()

    def set(self, other):
        """
        Set this transformation from the frozen copy of another
        :class:`Affine3DBase` object.
        """
        assert isinstance(other, Affine3DBase)
        self._mtx = other.get_matrix()
        self.invalidate()

    @staticmethod
    def identity():
        """
        (staticmethod) Return a new :class:`Affine3D` object that is
        the identity transform.

        Unless this transform will be mutated later on, consider using
        the faster :class:`IdentityTransform` class instead.
        """
        return Affine3D(np.identity(4))

    def clear(self):
        """
        Reset the underlying matrix to the identity transform.
        """
        self._mtx = np.identity(4)
        self.invalidate()
        return self

    def rotate(self, theta, phi=0):
        # TODO: Update docs!
        """
        Add a rotation (in radians) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # TODO: Update this!
        a = np.cos(theta)
        b = np.sin(theta)
        rotate_mtx = np.array(
            [[a, -b, 0.0], [b, a, 0.0], [0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(rotate_mtx, self._mtx)
        self.invalidate()
        return self

    def rotate_deg(self, degrees, phi=0):
        """
        Add a rotation (in degrees) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.rotate(degrees*np.pi/180., phi*np.pi/180.)

    # TODO: Need to figure these out.
    """
    def rotate_around(self, x, y, theta):
        """
        Add a rotation (in radians) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """
        Add a rotation (in degrees) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)
    """

    def translate(self, tx, ty, tz=0):
        """
        Adds a translation in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        translate_mtx = np.array(
            [[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, ty],
             [0.0, 0.0, 1.0, tz], [0.0, 0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(translate_mtx, self._mtx)
        self.invalidate()
        return self

    def scale(self, sx, sy=None, sz=None):
        # TODO: Update docs!
        """
        Adds a scale in place.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        If *sz* is None, the same scale is applied in both the *y*- and
        *z*-directions.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sy
        scale_mtx = np.array(
            [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0],
             [0.0, 0.0, sz, 0.0], [0.0, 0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(scale_mtx, self._mtx)
        self.invalidate()
        return self

    # TODO: I don't know, is it?
    def _get_is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0
    is_separable = property(_get_is_separable)


class IdentityTransform3D(IdentityTransform, Affine3DBase):
    """
    A special class that does one thing, the identity transform, in a
    fast way.
    """
    _mtx = np.identity(4)

    def __repr__(self):
        return "IdentityTransform3D()"
    __str__ = __repr__

class BlendedGenericTransform3D(BlendedGenericTransform, Transform):
    """
    A "blended" transform uses a transform for each direction.

    This "generic" version can handle any given child transform in the
    *x*, *y* and *z*-directions.
    """
    input_dims = 3
    output_dims = 3
    is_separable = True
    pass_through = True

    def __init__(self, x_transform, y_transform, z_transform):
        """
        Create a new "blended" transform using *x_transform* to
        transform the *x*-axis and *y_transform* to transform the
        *y*-axis and *z_transform* to transform the *z*-axis.

        You will generally not call this constructor directly but use
        the :func:`blended_transform_factory` function instead, which
        can determine automatically which kind of blended transform to
        create.
        """
        # Here we ask: "Does it blend?"

        self._z = z_transform
        BlendedGenericTransform3D.__init__(self, x_transform, y_transform)
        self.set_children(z_transform)

    def _get_is_affine(self):
        return BlendedGenericTransform._get_is_affine(self) and self._z.is_affine
    is_affine = property(_get_is_affine)

    def frozen(self):
        return blended_transform_factory(self._x.frozen(), self._y.frozen(), self._z.frozen())
    frozen.__doc__ = Transform.frozen.__doc__

    def __repr__(self):
        return "BlendedGenericTransform3D(%s,%s,%s)" % (self._x, self._y, self._z)
    __str__ = __repr__

    # TODO: Huh????????????????
    def transform(self, points):
        x = self._x
        y = self._y
        z = self._z

        if x is y and x.input_dims == 2:
            return x.transform(points)

        if x.input_dims == 2:
            x_points = x.transform(points)[:, 0:1]
        else:
            x_points = x.transform(points[:, 0])
            x_points = x_points.reshape((len(x_points), 1))

        if y.input_dims == 2:
            y_points = y.transform(points)[:, 1:]
        else:
            y_points = y.transform(points[:, 1])
            y_points = y_points.reshape((len(y_points), 1))

        if isinstance(x_points, MaskedArray) or isinstance(y_points, MaskedArray):
            return ma.concatenate((x_points, y_points), 1)
        else:
            return np.concatenate((x_points, y_points), 1)
    transform.__doc__ = Transform.transform.__doc__

    def transform_non_affine(self, points):
        if self.is_affine :
            return points
        return self.transform(points)
    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def inverted(self):
        return BlendedGenericTransform3D(self._x.inverted(), self._y.inverted(), self._z.inverted())
    inverted.__doc__ = Transform.inverted.__doc__

    def get_affine(self):
        if self._invalid or self._affine is None:
            if self.is_affine :
                # TODO: Huh??????????????????????
                if self._x == self._y:
                    self._affine = self._x.get_affine()
                else:
                    x_mtx = self._x.get_affine().get_matrix()
                    y_mtx = self._y.get_affine().get_matrix()
                    z_mtx = self._z.get_affine().get_matrix()
                    # This works because we already know the transforms are
                    # separable, though normally one would want to set ? and
                    # ? to zero.
                    mtx = np.vstack((x_mtx[0], y_mtx[1], z_mtx[2], [0.0, 0.0, 0.0, 1.0]))
                    self._affine = Affine3D(mtx)
            else:
                self._affine = IdentityTransform3D()
            self._invalid = 0
        return self._affine
    get_affine.__doc__ = Transform.get_affine.__doc__


class BlendedAffine3D(Affine3DBase):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This version is an optimization for the case where both child
    transforms are of type :class:`Affine2DBase`.
    """
    is_separable = True

    def __init__(self, x_transform, y_transform):
        """
        Create a new "blended" transform using *x_transform* to
        transform the *x*-axis and *y_transform* to transform the
        *y*-axis.

        Both *x_transform* and *y_transform* must be 2D affine
        transforms.

        You will generally not call this constructor directly but use
        the :func:`blended_transform_factory` function instead, which
        can determine automatically which kind of blended transform to
        create.
        """
        assert x_transform.is_affine
        assert y_transform.is_affine
        assert x_transform.is_separable
        assert y_transform.is_separable

        Transform.__init__(self)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)

        Affine2DBase.__init__(self)
        self._mtx = None

    def __repr__(self):
        return "BlendedAffine2D(%s,%s)" % (self._x, self._y)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()
            else:
                x_mtx = self._x.get_matrix()
                y_mtx = self._y.get_matrix()
                # This works because we already know the transforms are
                # separable, though normally one would want to set b and
                # c to zero.
                self._mtx = np.vstack((x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]))
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


def blended_transform_factory(x_transform, y_transform):
    """
    Create a new "blended" transform using *x_transform* to transform
    the *x*-axis and *y_transform* to transform the *y*-axis.

    A faster version of the blended transform is returned for the case
    where both child transforms are affine.
    """
    if (isinstance(x_transform, Affine2DBase)
        and isinstance(y_transform, Affine2DBase)):
        return BlendedAffine2D(x_transform, y_transform)
    return BlendedGenericTransform(x_transform, y_transform)


class CompositeGenericTransform(Transform):
    """
    A composite transform formed by applying transform *a* then
    transform *b*.

    This "generic" version can handle any two arbitrary
    transformations.
    """
    pass_through = True

    def __init__(self, a, b):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        You will generally not call this constructor directly but use
        the :func:`composite_transform_factory` function instead,
        which can automatically choose the best kind of composite
        transform instance to create.
        """
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        Transform.__init__(self)
        self._a = a
        self._b = b
        self.set_children(a, b)

    def frozen(self):
        self._invalid = 0
        frozen = composite_transform_factory(self._a.frozen(), self._b.frozen())
        if not isinstance(frozen, CompositeGenericTransform):
            return frozen.frozen()
        return frozen
    frozen.__doc__ = Transform.frozen.__doc__

    def _get_is_affine(self):
        return self._a.is_affine and self._b.is_affine
    is_affine = property(_get_is_affine)

    def _get_is_separable(self):
        return self._a.is_separable and self._b.is_separable
    is_separable = property(_get_is_separable)

    def __repr__(self):
        return "CompositeGenericTransform(%s, %s)" % (self._a, self._b)
    __str__ = __repr__

    def transform(self, points):
        return self._b.transform(
            self._a.transform(points))
    transform.__doc__ = Transform.transform.__doc__

    def transform_affine(self, points):
        return self.get_affine().transform(points)
    transform_affine.__doc__ = Transform.transform_affine.__doc__

    def transform_non_affine(self, points):
        if self._a.is_affine and self._b.is_affine:
            return points
        return self._b.transform_non_affine(
            self._a.transform(points))
    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def transform_path(self, path):
        return self._b.transform_path(
            self._a.transform_path(path))
    transform_path.__doc__ = Transform.transform_path.__doc__

    def transform_path_affine(self, path):
        return self._b.transform_path_affine(
            self._a.transform_path(path))
    transform_path_affine.__doc__ = Transform.transform_path_affine.__doc__

    def transform_path_non_affine(self, path):
        if self._a.is_affine and self._b.is_affine:
            return path
        return self._b.transform_path_non_affine(
            self._a.transform_path(path))
    transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__

    def get_affine(self):
        if self._a.is_affine and self._b.is_affine:
            return Affine2D(np.dot(self._b.get_affine().get_matrix(),
                                    self._a.get_affine().get_matrix()))
        else:
            return self._b.get_affine()
    get_affine.__doc__ = Transform.get_affine.__doc__

    def inverted(self):
        return CompositeGenericTransform(self._b.inverted(), self._a.inverted())
    inverted.__doc__ = Transform.inverted.__doc__


class CompositeAffine2D(Affine2DBase):
    """
    A composite transform formed by applying transform *a* then transform *b*.

    This version is an optimization that handles the case where both *a*
    and *b* are 2D affines.
    """
    def __init__(self, a, b):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        Both *a* and *b* must be instances of :class:`Affine2DBase`.

        You will generally not call this constructor directly but use
        the :func:`composite_transform_factory` function instead,
        which can automatically choose the best kind of composite
        transform instance to create.
        """
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims
        assert a.is_affine
        assert b.is_affine

        Affine2DBase.__init__(self)
        self._a = a
        self._b = b
        self.set_children(a, b)
        self._mtx = None

    def __repr__(self):
        return "CompositeAffine2D(%s, %s)" % (self._a, self._b)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            self._mtx = np.dot(
                self._b.get_matrix(),
                self._a.get_matrix())
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


def composite_transform_factory(a, b):
    """
    Create a new composite transform that is the result of applying
    transform a then transform b.

    Shortcut versions of the blended transform are provided for the
    case where both child transforms are affine, or one or the other
    is the identity transform.

    Composite transforms may also be created using the '+' operator,
    e.g.::

      c = a + b
    """
    if isinstance(a, IdentityTransform):
        return b
    elif isinstance(b, IdentityTransform):
        return a
    elif isinstance(a, AffineBase) and isinstance(b, AffineBase):
        return CompositeAffine2D(a, b)
    return CompositeGenericTransform(a, b)


class BboxTransform(Affine2DBase):
    """
    :class:`BboxTransform` linearly transforms points from one
    :class:`Bbox` to another :class:`Bbox`.
    """
    is_separable = True

    def __init__(self, boxin, boxout):
        """
        Create a new :class:`BboxTransform` that linearly transforms
        points from *boxin* to *boxout*.
        """
        assert boxin.is_bbox
        assert boxout.is_bbox

        Affine2DBase.__init__(self)
        self._boxin = boxin
        self._boxout = boxout
        self.set_children(boxin, boxout)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransform(%s, %s)" % (self._boxin, self._boxout)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            outl, outb, outw, outh = self._boxout.bounds
            x_scale = outw / inw
            y_scale = outh / inh
            if DEBUG and (x_scale == 0 or y_scale == 0):
                raise ValueError("Transforming from or to a singular bounding box.")
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale+outl)],
                                   [0.0    , y_scale, (-inb*y_scale+outb)],
                                   [0.0    , 0.0    , 1.0        ]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformTo(Affine2DBase):
    """
    :class:`BboxTransformTo` is a transformation that linearly
    transforms points from the unit bounding box to a given
    :class:`Bbox`.
    """
    is_separable = True

    def __init__(self, boxout):
        """
        Create a new :class:`BboxTransformTo` that linearly transforms
        points from the unit bounding box to *boxout*.
        """
        assert boxout.is_bbox

        Affine2DBase.__init__(self)
        self._boxout = boxout
        self.set_children(boxout)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransformTo(%s)" % (self._boxout)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            outl, outb, outw, outh = self._boxout.bounds
            if DEBUG and (outw == 0 or outh == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[outw,  0.0, outl],
                                   [ 0.0, outh, outb],
                                   [ 0.0,  0.0,  1.0]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformToMaxOnly(BboxTransformTo):
    """
    :class:`BboxTransformTo` is a transformation that linearly
    transforms points from the unit bounding box to a given
    :class:`Bbox` with a fixed upper left of (0, 0).
    """
    def __repr__(self):
        return "BboxTransformToMaxOnly(%s)" % (self._boxout)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            xmax, ymax = self._boxout.max
            if DEBUG and (xmax == 0 or ymax == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[xmax,  0.0, 0.0],
                                  [ 0.0, ymax, 0.0],
                                  [ 0.0,  0.0, 1.0]],
                                 np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformFrom(Affine2DBase):
    """
    :class:`BboxTransformFrom` linearly transforms points from a given
    :class:`Bbox` to the unit bounding box.
    """
    is_separable = True

    def __init__(self, boxin):
        assert boxin.is_bbox

        Affine2DBase.__init__(self)
        self._boxin = boxin
        self.set_children(boxin)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransformFrom(%s)" % (self._boxin)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            if DEBUG and (inw == 0 or inh == 0):
                raise ValueError("Transforming from a singular bounding box.")
            x_scale = 1.0 / inw
            y_scale = 1.0 / inh
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale)],
                                   [0.0    , y_scale, (-inb*y_scale)],
                                   [0.0    , 0.0    , 1.0        ]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class ScaledTranslation(Affine2DBase):
    """
    A transformation that translates by *xt* and *yt*, after *xt* and *yt*
    have been transformad by the given transform *scale_trans*.
    """
    def __init__(self, xt, yt, scale_trans):
        Affine2DBase.__init__(self)
        self._t = (xt, yt)
        self._scale_trans = scale_trans
        self.set_children(scale_trans)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "ScaledTranslation(%s)" % (self._t,)
    __str__ = __repr__

    def get_matrix(self):
        if self._invalid:
            xt, yt = self._scale_trans.transform_point(self._t)
            self._mtx = np.array([[1.0, 0.0, xt],
                                   [0.0, 1.0, yt],
                                   [0.0, 0.0, 1.0]],
                                  np.float_)
            self._invalid = 0
            self._inverted = None
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class TransformedPath(TransformNode):
    """
    A :class:`TransformedPath` caches a non-affine transformed copy of
    the :class:`~matplotlib.path.Path`.  This cached copy is
    automatically updated when the non-affine part of the transform
    changes.
    """
    def __init__(self, path, transform):
        """
        Create a new :class:`TransformedPath` from the given
        :class:`~matplotlib.path.Path` and :class:`Transform`.
        """
        assert isinstance(transform, Transform)
        TransformNode.__init__(self)

        self._path = path
        self._transform = transform
        self.set_children(transform)
        self._transformed_path = None
        self._transformed_points = None

    def _revalidate(self):
        if ((self._invalid & self.INVALID_NON_AFFINE == self.INVALID_NON_AFFINE)
            or self._transformed_path is None):
            self._transformed_path = \
                self._transform.transform_path_non_affine(self._path)
            self._transformed_points = \
                Path(self._transform.transform_non_affine(self._path.vertices),
                     None, self._path._interpolation_steps)
        self._invalid = 0

    def get_transformed_points_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.  Unlike
        :meth:`get_transformed_path_and_affine`, no interpolation will
        be performed.
        """
        self._revalidate()
        return self._transformed_points, self.get_affine()

    def get_transformed_path_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.
        """
        self._revalidate()
        return self._transformed_path, self.get_affine()

    def get_fully_transformed_path(self):
        """
        Return a fully-transformed copy of the child path.
        """
        if ((self._invalid & self.INVALID_NON_AFFINE == self.INVALID_NON_AFFINE)
            or self._transformed_path is None):
            self._transformed_path = \
                self._transform.transform_path_non_affine(self._path)
        self._invalid = 0
        return self._transform.transform_path_affine(self._transformed_path)

    def get_affine(self):
        return self._transform.get_affine()


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    '''
    Ensure the endpoints of a range are finite and not too close together.

    "too close" means the interval is smaller than 'tiny' times
    the maximum absolute value.

    If they are too close, each will be moved by the 'expander'.
    If 'increasing' is True and vmin > vmax, they will be swapped,
    regardless of whether they are too close.

    If either is inf or -inf or nan, return - expander, expander.
    '''
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return -expander, expander
    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True
    if vmax - vmin <= max(abs(vmin), abs(vmax)) * tiny:
        if vmin == 0.0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)

    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def interval_contains(interval, val):
    a, b = interval
    return (
        ((a < b) and (a <= val and b >= val))
        or (b <= val and a >= val))

def interval_contains_open(interval, val):
    a, b = interval
    return (
        ((a < b) and (a < val and b > val))
        or (b < val and a > val))

def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    '''
    Return a new transform with an added offset.
      args:
        trans is any transform
      kwargs:
        fig is the current figure; it can be None if units are 'dots'
        x, y give the offset
        units is 'inches', 'points' or 'dots'
    '''
    if units == 'dots':
        return trans + Affine2D().translate(x, y)
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')
    if units == 'points':
        x /= 72.0
        y /= 72.0
    elif not units == 'inches':
        raise ValueError('units must be dots, points, or inches')
    return trans + ScaledTranslation(x, y, fig.dpi_scale_trans)

