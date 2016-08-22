import requests
import requests_cache
import json
import time
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
from io import StringIO

# install request cache to limit calls to api while testing
requests_cache.install_cache('api_cache', backend='sqlite', expire_after=600)


class GetEnergy(object):
    """
    A class to capture an EIA API call
    """

    eia_url = 'http://api.eia.gov/series/'

    def __init__(self, api_key, series, freq=None, start=None, end=None):
        """
        Create eia_api object and related attriobutes from json
        :param api_key: an API key that is provided by EIA
        Optional parms required for date filter
        :param start: an optional start date as %Y-%m-%d %H:%M:%S
        :param end:an optional end date %Y%m%d %H
        :param freq:  frequency of data valid: 'A', 'M', 'W', 'D', 'H'
        """
        self.api_key = api_key
        self.series_id = series
        self.freq = freq
        self.start = self.format_date(self.freq, start)
        self.end = self.format_date(self.freq, end)
        self.json = self.get_series()
        self.df = CreateEnergyData(self.json).df

    def get_series(self):
        """
        Calls the EIA API with supplied api_key on init and series_id and return json
        """
        # default api params required for call
        api_parms = (
            ('api_key', self.api_key),
            ('series_id', self.series_id),
            ('start', self.start),
            ('end', self.end),
        )
        api_parms = tuple(i for i in api_parms if i[1] is not None)
        eia_req = requests.get(self.eia_url, params=api_parms)
        return json.loads(eia_req.text)

    @staticmethod
    def format_date(freq, date):
        """formats input dates to correct"""
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        freq_dict = {'A': '%Y', 'M': '%Y%m', 'W': '%Y%m%d',
                     'D': '%Y%m%d', 'H': '%Y%m%dT%HZ'}
        formatted_date = datetime.strftime(date, freq_dict[freq])
        return formatted_date


class GetWeatherForecast(object):
    """
    A class to capture an pyowm weather forecast call
    """

    owm_url = 'http://api.openweathermap.org/data/2.5/forecast'

    def __init__(self, api_key, city_id, units=None):
        """
        Create pyowm forecast object and return related attributes
        :param api_key: a valid Open Weath Map Api-Key
        :param city_id: a valid Open Weather Map city ID
        :param units: default is Kelvin, imperial=Fahrenheight
        """
        self.api_key = api_key
        self.city_id = city_id
        self.units = units
        self.json = self.get_series()
        self.df = CreateWeatherForecastData(self.json).df
        self.df_hr = InterpolateWeatherForecast(self.df).df

    def get_series(self):
        """
        Calls the EIA API with supplied api_key on init and series_id and return json
        """
        owm_parms = (
            ('id', self.city_id),
            ('APPID', self.api_key),
            ('units', self.units),
        )
        owm_req = requests.get(self.owm_url, params=owm_parms)
        return json.loads(owm_req.text)


class GetWeatherHistory(object):
    """
    A class to capture an pyowm weather history call
    """

    noaa_url = 'http://www7.ncdc.noaa.gov/rest/services/values/isd/{}/{}/{}/{}/?output=csv&token={}'

    def __init__(self, ncdc_api, station_id, variable, start, end):
        """
        Create pyowm forecast object and return related attributes
        :param api_key: a valid NCDC Api-Key
        :param station_id: a valid weather station
        :param variable: a ncdc recognized weather station variable
        :param start: a start time in %Y-%m-%d %H:%M:%S format
        :param end: and end time %Y-%m-%d %H:%M:%S format
        """
        self.api_key = ncdc_api
        self.station_id = station_id
        self.variable = variable
        self.start = start
        self.end = end
        self.req = self.get_series()
        self.df = CreateWeatheHistoryData(self.req).df

    def get_series(self):
        """
        Calls the EIA API with supplied api_key on init and series_id and return json
        """
        start = datetime.strftime(local_to_utc(
            datetime.strptime(self.start, '%Y-%m-%d %H:%M:%S')), '%Y%m%d%H%M')
        end = datetime.strftime(local_to_utc(
            datetime.strptime(self.end, '%Y-%m-%d %H:%M:%S')), '%Y%m%d%H%M')
        noaa_req = requests.get(self.noaa_url.format(self.station_id, self.variable, start,
                                                     end, self.api_key))
        return noaa_req


class CreateEnergyData(object):
    '''Creates the dataframe for Energy API call'''

    def __init__(self, json):
        """:param json: eia json"""
        self.json = json
        self.series = self.json['series']
        self.data = self.series[0]['data']
        self.df = self.create_dataframe()

    def create_dataframe(self):
        """Function to create dataframe from json['series'] """
        values = [x[1] for x in self.data]
        dates = self.get_dates()
        return pd.DataFrame(values, index=dates, columns=['values'])

    def get_dates(self):
        """Parses text dates to datetime index"""
        freq = {'A': '%Y', 'M': '%Y%m', 'W': '%Y%m%d',
                'D': '%Y%m%d', 'H': '%Y%m%d %H'}
        date_list = []
        for x in self.data:
            # need to add this ugly bit to remove hourly time format from EIA
            time = x[0].replace('T', ' ')
            time = time.replace('Z', '')
            date_list.append(datetime.strptime(
                time, freq[self.series[0]['f']]).strftime('%Y-%m-%d %H:%M:%S'))
        return date_list


class CreateWeatherForecastData(object):
    """Creates the dataframe for Open Weather Map API call"""

    def __init__(self, json):
        """":param json: a open weather map json object """
        self.json = json
        self.series = self.json['list']
        self.df = self.create_dataframe()

    def create_dataframe(self):
        """Function to create dataframe from json['list'] """
        dates = []
        values = []
        for i in self.series:
            time = datetime.strptime(i['dt_txt'], '%Y-%m-%d %H:%M:%S')
            dates.append(utc_to_local(time))
            values.append(i['main']['temp'])
        return pd.DataFrame(values, index=dates, columns=['temp'])


class CreateWeatheHistoryData(object):
    """Creates the dataframe for NCDC API call"""

    def __init__(self, req):
        """
        :param req: a request object returned from NCDC call
        """
        self.req = req
        self.df = self.create_dataframe().dropna(axis=1, how='all')

    def create_dataframe(self):
        """Function to create dataframe from req csv object"""
        req_df = pd.read_csv(StringIO(self.req.text), header=None,
                             na_values='null', keep_default_na=True, na_filter=True)
        return self.apply_filters(req_df)

    def apply_filters(self, df):
        # filter out non hourly increment reads
        df[df.iloc[:, 19] == 'FM-15']
        # pad hourly format
        df.iloc[:, 3] = df.iloc[:, 3].map("{:04}".format)
        df.loc[:, 'date'] = pd.to_datetime(
            df.iloc[:, 2].map(str) + df.iloc[:, 3].map(str), format='%Y%m%d%H%M')
        df.loc[:, 'date'] = df.loc[:, 'date'].map(utc_to_local)
        df.loc[:, 'temp'] = df.iloc[:, 5].map(self.convert_ncdc_temp)
        return df.set_index(df['date'])

    @staticmethod
    def convert_ncdc_temp(temp):
        """converts ncdc temp top F
        :param temp: and ncdc formatted temp
        """
        # ncdc temp is in Celsius *10 then convert to F
        return temp / 10 * 9 / 5 + 32


class InterpolateWeatherForecast(object):
    """Creates hourly forecast from OWM 3 hour forecast"""

    def __init__(self, forecast):
        """
        :param forecast: a weather forecast dataframe from GetWeatherForecast
        """
        self.forecast = forecast
        self.hours = self.time_hourly()
        self.df = self.interpolate()

    def time_hourly(self):
        """creates hourly time from 3 hour owm forecast"""
        min_time, max_time = self.forecast.index.min(), self.forecast.index.max()
        return pd.date_range(min_time, max_time, freq='H')

    def interpolate(self):
        """interpolate function suing np cubic interpolation"""
        date_len = len(self.forecast.index)
        date_axis = np.linspace(1, date_len, date_len, endpoint=True)
        f_cubic = interp1d(date_axis, self.forecast['temp'])
        temp_int = f_cubic(np.linspace(
            1, date_len, num=date_len * 3 - 2, endpoint=True))
        df = pd.DataFrame(temp_int, index=self.hours, columns=['temp'])
        return df


def utc_to_local(date):
    """converts from gmt to local"""
    epoch = time.mktime(date.timetuple())
    offset = datetime.fromtimestamp(
        epoch) - datetime.utcfromtimestamp(epoch)
    return date + offset


def local_to_utc(date):
    """converts from local to gmt"""
    epoch = time.mktime(date.timetuple())
    offset = datetime.fromtimestamp(
        epoch) - datetime.utcfromtimestamp(epoch)
    return date - offset
