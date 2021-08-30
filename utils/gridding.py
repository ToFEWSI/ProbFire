"""
@author: Tadas Nikonovas
helper functions, part of:
ProbFire, a probabilistic fire early warning system for
Indonesia.
"""

import numpy as np
import xarray as xr
import pandas as pd

#Define sub-region bounding boxes
bboxes = {
        'South Kalimantan': [-1.5, 110.5, -4., 115],
        'South Sumatra': [-2.2, 103, -5, 106.2],
        'South Papua': [-6, 137, -9., 142],
        'Central Sumatra': [2.6, 99.5, -2.2, 104.5],
        'West Kalimantan': [1., 108.8, -1.5, 112]
        }

def spatial_subset_dfr(dfr, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dfr - pandas dataframe
        bbox - (list) [North, West, South, East]
    Returns:
        pandas dataframe
    """
    dfr = dfr[(dfr['latitude'] < bbox[0]) &
                            (dfr['latitude'] > bbox[2])]
    dfr = dfr[(dfr['longitude'] > bbox[1]) &
                            (dfr['longitude'] < bbox[3])]
    return dfr

def lat_lon_grid_points(bbox, step):
    """
    Returns two lists with latitude and longitude grid cell center coordinates
    given the bbox and step.
    """
    lat_bbox = [bbox[0], bbox[2]]
    lon_bbox = [bbox[1], bbox[3]]
    latmin = lat_bbox[np.argmin(lat_bbox)]
    latmax = lat_bbox[np.argmax(lat_bbox)]
    lonmin = lon_bbox[np.argmin(lon_bbox)]
    lonmax = lon_bbox[np.argmax(lon_bbox)]
    numlat = int((latmax - latmin) / step) + 1
    numlon = int((lonmax - lonmin) / step) + 1
    lats = np.linspace(latmin, latmax, numlat, endpoint = True)
    lons = np.linspace(lonmin, lonmax, numlon, endpoint = True)
    return lats, lons

class Gridder(object):
    def __init__(self, lats=None, lons=None, bbox=None, step=None):
        if all(cord is not None for cord in [lats, lons]):
            self.lats, self.lons = lats, lons
            self.step = self.grid_step()
        elif all(item is not None for item in [bbox, step]):
            self.step = step
            if isinstance(bbox, list):
                self.lats, self.lons = lat_lon_grid_points(bbox, step)
            if isinstance(bbox, str):
                self.lats, self.lons = lat_lon_grid_points(self.bboxes[bbox], step)
        else:
            print('Please provide either lats + lons or bbox + step')
            return None
        self.bbox = self.grid_bbox()
        self.grid_bins()

    def grid_step(self):
        return (self.lons[1] - self.lons[0])

    def grid_bbox(self):
        lat_min, lat_max = self.lats.min(), self.lats.max()
        lon_min, lon_max = self.lons.min(), self.lons.max()
        self.lat_min = lat_min - self.step * 0.5
        self.lon_min = lon_min - self.step * 0.5
        if lat_max < 0:
            self.lat_max = lat_max + self.step * 0.5
        else:
            self.lat_max = lat_max - self.step * 0.5
        self.lon_max = lon_max + self.step * 0.5
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]

    def grid_bins(self):
        self.lon_bins = np.arange(self.lon_min, self.lon_max, self.step)
        self.lat_bins = np.arange(self.lat_min, self.lat_max, self.step)

    def binning(self, lon, lat):
        """
        Get indices of the global grid bins for the longitudes and latitudes
        of observations stored in frpFrame pandas DataFrame. Must have 'lon' and 'lat'
        columns.

        Arguments:
            lon : np.array, representing unprojected longitude coordinates.
            lat : np.array, representing unprojected longitude coordinates.

        Returns:
            Raises TypeError if frpFrame is not a pandas DataFrame
            frpFrame : pandas DataFrame
                Same DataFrame with added columns storing positional indices
                in the global grid defined in grid_bins method
        """
        lonind = np.digitize(lon, self.lon_bins) - 1
        latind = np.digitize(lat, self.lat_bins) - 1
        return lonind, latind

    def add_grid_inds(self, dfr):
        lonind, latind = self.binning(dfr['longitude'].values, dfr['latitude'].values)
        dfr.loc[:, 'lonind'] = lonind
        dfr.loc[:, 'latind'] = latind
        return dfr

    def add_coords_from_ind(self, dfr):
        """
        Add longitude and latitude columns to the input dfr
        based on lonind and latind columns
        """
        dfr['longitude'] = self.lons[dfr.lonind]
        dfr['latitude'] = self.lats[dfr.latind]
        return dfr

    def calc_area(self, dfr):
        """
        Calculate area of the grid cells in the dataframe besed on
        the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1 = np.deg2rad(0)
        lon2 = np.deg2rad(self.step)
        # convert decimal degrees to radians 
        latitude = self.lats[dfr.latind]
        lat1 = np.deg2rad(latitude)
        # haversine formula 
        dlon = lon2 - lon1
        dist_lat = dist_on_earth(0, dlon, 0, 0)
        dist_lon = dist_on_earth(dlon, 0, lat1, lat1)
        area = dist_lat * dist_lon
        dfr['cell_area'] = area
        return dfr

    def spatial_subset_ind_dfr(self, dfr, bbox):
        """
        Selects data within spatial bbox. bbox coords must be given as
        positive values for the Northern hemisphere, and negative for
        Southern. West and East both positive - Note - the method is
        naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
        Args:
            dfr - pandas dataframe
            bbox - (list) [North, West, South, East]
        Returns:
            pandas dataframe
        """
        sbox = np.where((self.lats < bbox[0]) & (self.lats > bbox[2]))
        ebox = np.where((self.lons > bbox[1]) & (self.lons < bbox[3]))
        dfr = dfr[(dfr['latind'] <= sbox[0].max()) &
                                (dfr['latind'] >= sbox[0].min())]
        dfr = dfr[(dfr['lonind'] >= ebox[0].min()) &
                                (dfr['lonind'] <= ebox[0].max())]
        return dfr

if __name__ == '__main__':
    pass

