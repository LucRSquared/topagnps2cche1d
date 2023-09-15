import numpy as np
import holoviews as hv


class CrossSection:
    def __init__(
        self,
        id,
        ws=[],
        zs=[],
        lfp_idx=None,
        lbt_idx=None,
        rbt_idx=None,
        rfp_idx=None,
        mannings_roughness=0.03,
        **kwargs,
    ):
        """
        Define a cross section in the local transverse plane coordinates along the channel. The coordinates are given in the left to right
        format (when looking downstream)
        Arguments:
        * id: cross section unique id
        * ws: cross section abscissas arrays from left to right
        * zs: cross section ordinates arrays
        * lfp_idx: 'Left Flood Plain' edge Index of cross section marking the beginning of the main channel
        * lbt_idx: 'Left Bank Toe' Index of cross section marking the toe of the left bank
        * rbt_idx: 'Right Bank Toe'
        * rfp_idx: 'Right Flood Plain' Index marking the end of the main channel on the right bank
        /!\ these indexes start at 1 (ONE) not 0 (ZERO)
        * mannings_roughness : Manning's Roughness coefficient
            Either a float representing the roughness of the entire cross section
            or an array of length len(zs) where n_rgh[i] is the roughness of point (ws[i], zs[i])
            # FUN FACT: Did you know that the Manning's Roughness coefficient was first invented by
            # french engineer Philippe Gaspard Gauckler in 1868 and later re-developed by Robert Manning in 1890?
            # So really this should be the Gauckler roughness coefficient but no one's holding any grudges ;-)
        Key-Value Arguments:
        * type:
            - 'trapezoidal_with_flood_plain' which accepts the parameters defined below
                * 'bottom_elevation'          : elevation of bottom of channel
                * 'flood_plain_width'         : width of the flood plain on each side
                * 'main_channel_bottom_width' : width of the bottom of the main channel
                * 'main_channel_depth'        : depth of main channel measured from bottom to elevation of flood plain
                * 'main_channel_bank_slope'   : slope of main channel banks
                * 'flood_plain_bank_slope'    : slope of flood plain banks
                * 'flood_plain_bank_height    : height of flood plain bank
                If these parameters are not provided default values will be used!
            - 'user_defined' : the user provides the coordinates ws, zs, and lfp_idx, lbt_idx, rbt_idx, rfp_idx
                if the indexes are not provided then they are defaulted to end and beginning indexes

        """

        self.id = id
        self.ws = np.array(ws)
        self.zs = np.array(zs)
        self.lfp_idx = lfp_idx
        self.rfp_idx = rfp_idx
        self.lbt_idx = lbt_idx
        self.rbt_idx = rbt_idx
        self.n_rgh = mannings_roughness

        if "type" in kwargs:
            xs_type = kwargs["type"]
        elif self.ws.any():
            xs_type = "user_defined"
        else:
            xs_type = "trapezoidal_with_flood_plain"

        if xs_type == "trapezoidal_with_flood_plain":
            (
                self.ws,
                self.zs,
                self.lfp_idx,
                self.lbt_idx,
                self.rbt_idx,
                self.rfp_idx,
            ) = self.define_trapz_cross_section_width_slope_elevation(**kwargs)

        self._assert_valid_cross_section()

    def _assert_valid_cross_section(self):
        """
        Make sure the provided cross-section is valid
        """
        ws, zs = self.ws, self.zs
        numpoints = len(ws)
        lfp, lbt, rbt, rfp = self.lfp_idx, self.lbt_idx, self.rbt_idx, self.rfp_idx
        if ws.size == 0 or zs.size == 0:
            raise AssertionError("Empty cross section coordinates")
        elif ws.size != zs.size:
            raise AssertionError("Ws and Zs coordinates should be the same length")
        elif any([item is None for item in [lfp, rfp, lbt, rbt]]):
            self.lfp_idx, self.lbt_idx, self.rbt_idx, self.rfp_idx = (
                1,
                1,
                numpoints,
                numpoints,
            )
            lfp, lbt, rbt, rfp = self.lfp_idx, self.lbt_idx, self.rbt_idx, self.rfp_idx
        elif not (0 <= lfp <= lbt < rbt <= rfp <= zs.size):
            raise AssertionError(
                "Invalid LFP, LBT, RBT, RFP index values: maybe out of bounds or incorrect order?"
            )

        # Making sure Manning's Roughness coefficient
        n_rgh = self.n_rgh
        if isinstance(n_rgh, float):
            self.n_rgh = np.full(numpoints, n_rgh)

        elif len(n_rgh) != numpoints:
            raise AssertionError(
                "Invalid 'mannings_roughness', should be a float or array of length numpoints"
            )

    def shift_elevations(self, k):
        """
        Add k to elevation cross section
        """
        self.zs = self.zs + k

    def get_thalweg_elevation(self):
        """
        Get elevation of the lowest point of the cross-section
        """
        return self.zs.min()
    
    def set_thalweg_elevation(self, thalweg_z):
        """
        Shift elevation so that the cross section's Thalweg elevation is the one specified
        """
        old_thalweg_z = self.get_thalweg_elevation()
        self.zs = self.zs - old_thalweg_z + thalweg_z

    @classmethod
    def define_trapz_cross_section_width_slope_elevation(cls, **kwargs):
        """
        A trapezoidal cross-section has 8 points:
           3 points for the left flood plain (a bank + a flat part)
        +  3 points for the right flood plain (a flat part + bank)
        +  2 points defining the bottom of the main channel

        Key-Value arguments:
        * 'bottom_elevation'          : elevation of bottom of channel
        * 'flood_plain_width'         : width of the flood plain on each side
        * 'main_channel_bottom_width' : width of the bottom of the main channel
        * 'main_channel_depth'        : depth of main channel measured from bottom to elevation of flood plain
        * 'main_channel_bank_slope'   : slope of main channel banks
        * 'flood_plain_bank_slope'    : slope of flood plain banks
        * 'flood_plain_bank_height    : height of flood plain bank
        """
        if "bottom_elevation" in kwargs:
            bottom_elevation = kwargs["bottom_elevation"]
        else:
            bottom_elevation = 0

        if "flood_plain_width" in kwargs:
            flood_plain_width = kwargs["flood_plain_width"]
            if flood_plain_width <= 0:
                raise AssertionError("'flood_plain_width' should be a positive number")
        else:
            flood_plain_width = 22

        if "main_channel_bottom_width" in kwargs:
            main_channel_bottom_width = kwargs["main_channel_bottom_width"]
            if main_channel_bottom_width <= 0:
                raise AssertionError(
                    "'main_channel_bottom_width' should be a positive number"
                )
        else:
            main_channel_bottom_width = 20

        if "main_channel_depth" in kwargs:
            main_channel_depth = kwargs["main_channel_depth"]
            if main_channel_depth <= 0:
                raise AssertionError("'main_channel_depth' should be a positive number")
        else:
            main_channel_depth = 2

        if "main_channel_bank_slope" in kwargs:
            main_channel_bank_slope = kwargs["main_channel_bank_slope"]
            if main_channel_bank_slope <= 0:
                raise AssertionError(
                    "'main_channel_bank_slope' should be a positive number"
                )
        else:
            main_channel_bank_slope = 2 / 3

        if "flood_plain_bank_slope" in kwargs:
            flood_plain_bank_slope = kwargs["flood_plain_bank_slope"]
            if flood_plain_bank_slope <= 0:
                raise AssertionError(
                    "'flood_plain_bank_slope' should be a positive number"
                )
        else:
            flood_plain_bank_slope = 0.5

        if "flood_plain_bank_height" in kwargs:
            flood_plain_bank_height = kwargs["flood_plain_bank_height"]
            if flood_plain_bank_height <= 0:
                raise AssertionError(
                    "'flood_plain_bank_height' should be a positive number"
                )
        else:
            flood_plain_bank_height = 4

        zs = np.array(
            [
                main_channel_depth + flood_plain_bank_height,
                main_channel_depth,
                main_channel_depth,
                0,
                0,
                main_channel_depth,
                main_channel_depth,
                main_channel_depth + flood_plain_bank_height,
            ]
        )
        zs = zs + bottom_elevation

        half_chanel_w = np.array(
            [
                main_channel_bottom_width / 2,
                main_channel_depth / main_channel_bank_slope,
                flood_plain_width,
                flood_plain_bank_height / flood_plain_bank_slope,
            ]
        )
        half_chanel_w = half_chanel_w.cumsum()
        ws = np.r_[-half_chanel_w[::-1], half_chanel_w]

        lfp_idx, lbt_idx, rbt_idx, rfp_idx = 2, 3, 4, 5

        return ws, zs, lfp_idx, lbt_idx, rbt_idx, rfp_idx

    def plot(self, **kwargs):
        """
        Plot cross section
        """
        if "title" in kwargs:
            title = kwargs["title"]
        else:
            title = ""

        if "aspect" in kwargs:
            aspect = kwargs["aspect"]
        else:
            aspect = None

        if "frame_width" in kwargs:
            frame_width = kwargs["frame_width"]
        else:
            frame_width = 800

        if "frame_height" in kwargs:
            frame_height = kwargs["frame_height"]
        else:
            frame_height = 500

        if "line_width" in kwargs:
            line_width = kwargs["line_width"]
        else:
            line_width = 2

        if "marker" in kwargs:
            marker = kwargs["marker"]
        else:
            marker = "o"

        if "marker_size" in kwargs:
            marker_size = kwargs["marker_size"]
        else:
            marker_size = 7

        xs_plot = hv.Curve((self.ws, self.zs)).opts(line_width=line_width) * hv.Scatter(
            (self.ws, self.zs)
        ).opts(
            marker=marker,
            size=marker_size,
            frame_width=frame_width,
            frame_height=frame_height,
            aspect=aspect,
            title=title,
        )
        return xs_plot
