import numpy as np
import xarray as xr
import pandas as pd

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

def dist_on_earth(dlon, dlat, lat1, lat2):
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        kmeters = 6371 * c
        return kmeters

def to_day_since(dtime_string):
    """
    Method returning day since the self base date. Takes string datetime in
    YYYY-MM-DD format.
    """
    dtime = pd.to_datetime(dtime_string, format='%Y-%m-%d')
    return (dtime - self.basedate).days

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
        self.bboxes = {'indonesia': [8.0, 93.0, -13.0, 143.0],
                       'riau': [3, 99, -2, 104],
                       'canada': [83, -141, 41.67, -52.6],
                       'canada_usa': [83, -171.8, 20.67, -52.6]}
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


    def primary_to_grid(self, dfr):
        dfr = self.add_grid_inds(dfr)
        grouped = dfr.groupby(['lonind', 'latind', 'primary']).size().unstack(fill_value = 0)
        grouped.loc[:, 'total'] = grouped.sum(axis = 1)
        classes = grouped.columns.values
        print(classes)
        grouped.reset_index(inplace = True)
        dss = []
        for item in classes:
            print(item)
            gridded = self.dfr_to_grid(grouped[['lonind', 'latind', item]], item)
            dataset = xr.Dataset({str(item): (['latitude', 'longitude'], np.flipud(gridded))},
                                  coords={'latitude': self.lats,
                                         'longitude': self.lons})
            dss.append(dataset)
        return xr.merge(dss)


    def lulc_to_grid(self, dfr):
        dfr = self.add_grid_inds(dfr)
        grouped = dfr.groupby(['lonind', 'latind', 'lulc']).size().unstack(fill_value = 0)
        grouped.loc[:, 'total'] = grouped.sum(axis = 1)
        classes = grouped.columns.values
        print(classes)
        grouped.reset_index(inplace = true)
        dss = []
        for item in classes:
            print(item)
            gridded = self.dfr_to_grid(grouped[['lonind', 'latind', item]], item)
            dataset = xr.Dataset({str(item): (['latitude', 'longitude'], np.flipud(gridded))},
                                  coords={'latitude': self.lats,
                                         'longitude': self.lons})
            dss.append(dataset)
        return xr.merge(dss)

    def grid_array_to_dataset(self, grid_array, name):
        dataset = xr.Dataset({str(name): (['latitude', 'longitude'], np.flipud(grid_array))},
                              coords={'latitude': self.lats,
                                     'longitude': self.lons})
        return dataset

    def dfr_to_dataset(self, dfr, name, no_value):
        gridded = self.dfr_to_grid(dfr, name, no_value)
        dataset = self.grid_array_to_dataset(gridded, name)
        return dataset

    def to_grid(self, dfr):
        dfr = self.add_grid_inds(dfr)
        grouped = pd.DataFrame({'count' : dfr.groupby(['lonind', 'latind']).size()}).reset_index()
        gridded = self.dfr_to_grid(grouped, 'count', np.nan)
        return gridded

    def dfr_to_grid(self, dfr, column, no_value):
        gridded = np.empty((self.lats.shape[0],
                         self.lons.shape[0]))
        gridded[:, :] = no_value
        latinds = dfr['latind'].values.astype(int)
        loninds = dfr['lonind'].values.astype(int)
        gridded[latinds, loninds] = dfr[column]
        gridded = np.flipud(gridded)
        return gridded

    def to_xarray(self, data, var_name, timestamps):
        lats = np.arange((-90 + self.step / 2.), 90., self.step)[::-1]
        lons = np.arange((-180 + self.step / 2.), 180., self.step)
        dataset = xr.Dataset({var_name: (['latitude', 'longitude', 'date'], data)},
                              coords={'latitude': lats,
                                     'longitude': lons,
                                     'date': timestamps})
        return dataset

    def grid_dfr(self, dfr):
        dates = pd.date_range(dfr.date.min(), dfr.date.max(), freq='D')
        lonind, latind = self.binning(dfr['longitude'].values, dfr['latitude'].values)
        dfr.loc[:, 'lonind'] = lonind
        dfr.loc[:, 'latind'] = latind
        dfa = pd.DataFrame({'count': dfr.groupby(['date', 'latind', 'lonind']).size()})
        dfa = dfa.reset_index()
        dfa.loc[:, 'dind'] = (dfa.date - dfa.date.min()).dt.days
        grids = np.zeros((self.lats.shape[0], self.lons.shape[0], dfa.dind.max() + 1),
                         dtype = int)
        grids[dfa.latind, dfa.lonind, dfa.dind] = dfa['count'].astype(int)
        grids = np.flip(grids, axis = 0)
        dataset = xr.Dataset({'count': (['latitude', 'longitude', 'time'], grids)},
                              coords={'latitude': self.lats,
                                      'longitude': self.lons,
                                      'time': dates})
        return dataset


    def grid_centroids(self, years, dfr_list, distance):
            dsy = []
            for nr, dur in enumerate(['2', '4', '8', '16']):
                dfr = dfr_list[nr]
                gr_year = dfr[dfr.year == year]
                grouped_days = gr_year.groupby('date')
                grids = [self.to_grid(x[1]) for x in grouped_days]
                timestamps = [x[0] for x in grouped_days]
                netcdf_store = self.to_xarray(np.dstack(grids), 'ign_agg_{0}'.format(dur), timestamps)
                dsy.append(netcdf_store)
            dsa = xr.merge(dsy)
            dsa.to_netcdf('/mnt/data/frp/ignitions_tropics_{0}_{1}_frp.nc'.format(year, distance),
                          encoding={'ign_agg_2': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_4': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_8': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_16': {'dtype': 'int16', 'zlib': True}})

    def add_coords_from_ind(self, dfr):
        dfr['longitude'] = self.lons[dfr.lonind]
        dfr['latitude'] = self.lats[dfr.latind]
        return dfr

if __name__ == '__main__':
    pass

