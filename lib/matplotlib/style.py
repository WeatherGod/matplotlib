"""
Class for compartmentalizing the most common artist attributes that
impact the appearance of an artist.
"""
from itertools import cycle
from matplotlib.cbook import iterable

class Style(object) :
    #__all__ = ('color', 'edgecolor', 'facecolor', 'linestyle',
    #           'linewidth', 'marker', 'markeredgecolor', 'markeredgewidth',
    #           'markerfacecolor', 'markerfacecoloralt', 'markersize')

    def __init__(self, c=None, ec=None, fc=None,
                       hatch=None, ls=None, lw=None,
                       marker=None, mec=None, mew=None,
                       mfc=None, mfca=None, ms=None) :
        self._propcycle = dict()
        self._propval = dict()
        # TODO: maybe should be static?
        self._fallbacks = dict(ec="c", fc="c",
                               mec="ec", mew="lw", mfc="fc",
                               mfca="mfc")

        self.c = c
        self.ec = ec
        self.fc = fc
        self.hatch = hatch
        self.ls = ls
        self.lw = lw
        self.marker = marker
        self.mec = mec
        self.mew = mew
        self.mfc = mfc
        self.mfca = mfca
        self.ms = ms

    def _setter(self, propname, vals) :
        """
        General setter function for any property.

        *propname*      name of the property
        *vals*          the vals for the property
        """
        if vals is not None :
            if not iterable(vals) :
                vals = [vals]
            self._propcycle[propname] = cycle(vals)
        else :
            # Making sure that the propcycle entry is cleared with a None
            self._propcycle[propname] = vals
        # Resetting the propval entry
        self._propval[propname] = None

    def _getter(self, propname) :
        """
        General getter function for any property.
        This function will advance the cycle for the stated property.

        *propname*      name of the property
        """
        propval = self._propval.get(propname, None)
        if propval is not None :
            return propval

        propcycle = self._propcycle.get(propname, None)
        if propcycle is None :
            # There was no property cycle set for this property,
            # see if there is a fallback
            propval = None
            fallback = self._fallbacks.get(propname, None)
            if fallback is not None :
                # Perform the access of the property named *fallback*
                propval = self.__getattribute__(fallback)
            # Since it isn't its own value, don't save it into propval.
            # This way, if the cycle for the fallback gets changed, the
            # next access to this property will get the new fallback value.
            return propval
        else :
            # The property val has not been set, so it is free to advance
            # the property cycle.
            propval = next(self._propcycle[propname])
            self._propval[propname] = propval        
            return propval

    def unlock(self) :
        # Completely reset the cached property values
        self._propval = dict()

    #####################
    # Color
    #####################
    def _set_color(self, c) :
        self._setter("c", c)

    def _get_color(self) :
        return self._getter("c")

    c = property(_get_color, _set_color, None, "any matplotlib color")
    color = property(_get_color, _set_color, None, c.__doc__)
    colors = property(_get_color, _set_color, None, c.__doc__)

    def _set_edgecolor(self, ec) :
        self._setter("ec", ec)

    def _get_edgecolor(self) :
        return self._getter("ec")

    ec = property(_get_edgecolor, _set_edgecolor, None, "any matplotlib color")
    edgecolor = property(_get_edgecolor, _set_edgecolor, None, ec.__doc__)
    edgecolors = property(_get_edgecolor, _set_edgecolor, None, ec.__doc__)


    def _set_markeredgecolor(self, mec) :
        self._setter("mec", mec)

    def _get_markeredgecolor(self) :
        return self._getter("mec")

    mec = property(_get_markeredgecolor, _set_markeredgecolor,
                   None, "any matplotlib color")
    markeredgecolor = property(_get_markeredgecolor, _set_markeredgecolor,
                               None, mec.__doc__)
    markeredgecolors = property(_get_markeredgecolor, _set_markeredgecolor,
                                None, mec.__doc__)

    def _set_facecolor(self, fc) :
        self._setter("fc", fc)

    def _get_facecolor(self) :
        return self._getter("fc")

    fc = property(_get_facecolor, _set_facecolor, None, "any matplotlib color")
    facecolor = property(_get_facecolor, _set_facecolor, None, fc.__doc__)
    facecolors = property(_get_facecolor, _set_facecolor, None, fc.__doc__)

    def _set_markerfacecolor(self, mfc) :
        self._setter("mfc", mfc)

    def _get_markerfacecolor(self) :
        return self._getter("mfc")

    mfc = property(_get_markerfacecolor, _set_markerfacecolor,
                   None, "any matplotlib color")
    markerfacecolor = property(_get_markerfacecolor, _set_markerfacecolor,
                               None, mfc.__doc__)
    markerfacecolors = property(_get_markerfacecolor, _set_markerfacecolor,
                                None, mfc.__doc__)

    def _set_markerfacecoloralt(self, mfca) :
        self._setter("mfca", mfca)

    def _get_markerfacecoloralt(self) :
        return self._getter("mfca")

    mfca = property(_get_markerfacecoloralt, _set_markerfacecoloralt,
                    None, "any matplotlib color")
    markerfacecoloralt = property(_get_markerfacecoloralt,
                                  _set_markerfacecoloralt,
                                  None, mfca.__doc__)


    ###################
    # Hatch
    ###################
    def _set_hatch(self, hatch) :
        self._setter("hatch", hatch)

    def _get_hatch(self) :
        return self._getter("hatch")

    hatch = property(_get_hatch, _set_hatch, None, "hatching pattern")

    ###################
    # Linestyle
    ###################
    def _set_linestyle(self, ls) :
        self._setter("ls", ls)

    def _get_linestyle(self) :
        return self._getter("ls")

    ls = property(_get_linestyle, _set_linestyle, None, "linestyle pattern")
    linestyle = property(_get_linestyle, _set_linestyle, None, ls.__doc__)
    linestyles = property(_get_linestyle, _set_linestyle, None, ls.__doc__)
    # ????
    dashes = property(_get_linestyle, _set_linestyle, None, ls.__doc__)

    ###################
    # Width
    ###################
    def _set_linewidth(self, lw) :
        self._setter("lw", lw)

    def _get_linewidth(self) :
        return self._getter("lw")

    lw = property(_get_linewidth, _set_linewidth, None, "linewidth")
    linewidth = property(_get_linewidth, _set_linewidth, None, lw.__doc__)
    linewidths = property(_get_linewidth, _set_linewidth, None, lw.__doc__)


    def _set_markeredgewidth(self, mew) :
        self._setter("mew", mew)

    def _get_markeredgewidth(self) :
        return self._getter("mew")

    mew = property(_get_markeredgewidth, _set_markeredgewidth,
                   None, "marker edgewidth")
    markeredgewidth = property(_get_markeredgewidth, _set_markeredgewidth,
                               None, mew.__doc__)
    markeredgewidths = property(_get_markeredgewidth, _set_markeredgewidth,
                                None, mew.__doc__)



    ###################
    # Marker
    ###################
    def _set_marker(self, marker) :
        self._setter("marker", marker)

    def _get_marker(self) :
        return self._getter("marker")

    marker = property(_get_marker, _set_marker, None, "plot marker")

    def _set_markersize(self, ms) :
        self._setter("ms", ms)

    def _get_markersize(self) :
        return self._getter("ms")

    ms = property(_get_markersize, _set_markersize, None, "marker size")
    markersize = property(_get_markersize, _set_markersize, None, ms.__doc__)
    markersizes = property(_get_markersize, _set_markersize, None, ms.__doc__)

