import requests
import requests_cache
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
from io import StringIO
import time

# TODO GetEnergy can handle time series filter
# need to limit GetWeatherForecast to filter 24 hours df[:24]
# need to convert Weather to HDD/CDD
# write script to fetch Energy and Weather history from last week
# need to run a regression and use parms

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
        self.dataframe = CreateEnergyData(self.json).dataframe

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
        self.dataframe = CreateWeatherForecastData(self.json).dataframe
        #TODO create class for interpolation
        # self.hourly_time = self.get_time_hourly(self.dataframe)
        # self.hourly_temps = self.interpolate_weather(self.dataframe,
        #                                              self.hourly_time, 'temp')

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

    def interpolate_weather(self, weather_detail, time, key):
        """
        Interpolates the 3hr forecast to an hourly forecast using cubic interpolation
        :param weather_detail is dataframe weather attributes returned from get_weather_detail
        :param key is weather detail dataframe column (temp, temp_max or temp_min)
        """
        date_len = len(weather_detail.index)
        date_axis = np.linspace(1, date_len, date_len, endpoint=True)
        f_cubic = interp1d(date_axis, weather_detail[key])
        temp_int = f_cubic(np.linspace(
            1, date_len, num=date_len * 3 - 2, endpoint=True))
        df = pd.DataFrame(temp_int, index=time, columns=['temp'])
        df['hdd'] = df.apply(create_hdd, axis=1)
        df['cdd'] = df.apply(create_cdd, axis=1)
        return df

    def get_time_hourly(self, weather_detail):
        """
        Parses temperature and time from weather objects imbedded in forecast object
        :param weather detail is dictionary returned from get_weather_detail
        """
        min_time, max_time = weather_detail.index.min(), weather_detail.index.max()
        return pd.date_range(min_time, max_time, freq='H')


class GetWeatherHistory(object):
    """
    A class to capture an pyowm weather history call
    """

    noaa_url = 'http://www7.ncdc.noaa.gov/rest/services/values/isd/{}/{}/{}/{}/?output=csv&token={}'

    def __init__(self, ncdc_api, station_id, variable, start, end):
        """
        Create pyowm forecast object and return related attributes
        :param api_key: a valid Open Weath Map Api-Key
        :param city_id: a valid Open Weather Map city ID
        :param type: a time interval ex:hour
        :param units: default is Kelvin, imperial=Fahrenheight
        :param start: a start time in %Y-%m-%d %H:%M:%S format
        :param end: and end time %Y-%m-%d %H:%M:%S format
        """
        self.api_key = ncdc_api
        self.station_id = station_id
        self.variable = variable
        self.start = self.local2utc(start)
        self.end = self.local2utc(end)
        self.req = self.get_series()
        self.dataframe = self.create_dataframe().dropna(axis=1, how='all')

    def get_series(self):
        """
        Calls the EIA API with supplied api_key on init and series_id and return json
        """
        noaa_req = requests.get(self.noaa_url.format(self.station_id, self.variable, self.start,
                                                     self.end, self.api_key))
        return noaa_req

    def create_dataframe(self):
        df = pd.read_csv(StringIO(self.req.text), header=None,
                         na_values='null', keep_default_na=True, na_filter=True)
        # filter for hourly reads 'FM-15'
        df = df[df[19] == 'FM-15']
        df[3] = df[3].map("{:04}".format)
        df['date'] = pd.to_datetime(
            df[2].map(str) + df[3].map(str), format='%Y%m%d%H%M')
        df['date'] = df.apply(self.utc2local, axis=1)
        df['temp'] = df.apply(self.temp_convert, axis=1)
        df['hdd'] = df.apply(create_hdd, axis=1)
        df['cdd'] = df.apply(create_cdd, axis=1)
        return df.set_index(df['date'])

    @staticmethod
    def utc2local(date):
        epoch = time.mktime(date['date'].timetuple())
        offset = datetime.fromtimestamp(
            epoch) - datetime.utcfromtimestamp(epoch)
        return date['date'] + offset

    @staticmethod
    def local2utc(date):
        """Converts a time in local time to GMT to input to ncdc url
        :param date: a date in %Y-%m-%d %H:%M format
        """
        date = datetime.strptime(date, '%Y-%m-%d %H:%M')
        epoch = time.mktime(date.timetuple())
        offset = datetime.fromtimestamp(
            epoch) - datetime.utcfromtimestamp(epoch)
        date_gmt = datetime.strftime(date - offset, '%Y%m%d%H%M')
        return date_gmt

    @staticmethod
    def temp_convert(temp):
        return temp[5] / 10 * 9 / 5 + 32


def create_hdd(temp):
    """Converts F to HDD"""
    return max(0, 65 - temp['temp'])


def create_cdd(temp):
    """Converts F to DD"""
    return max(0, temp['temp'] - 65)


def local2utc(date):
    date = datetime.strptime(date, '%Y-%m-%d %H:%M')
    epoch = time.mktime(date.timetuple())
    offset = datetime.fromtimestamp(
        epoch) - datetime.utcfromtimestamp(epoch)
    date_gmt = datetime.strftime(date - offset, '%Y%m%d%H%M')
    return date_gmt


class CreateEnergyData(object):
    '''Creates the dataframe for Energy API call'''

    def __init__(self, json):
        """:param json: eia json"""
        self.json = json
        self.series = self.json['series']
        self.data = self.series[0]['data']
        self.dataframe = self.create_dataframe()

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
        self.dataframe = self.create_dataframe()

    def create_dataframe(self):
        """Function to create dataframe from json['list'] """
        dates = []
        values = []
        for i in self.series:
            time = datetime.strptime(i['dt_txt'], '%Y-%m-%d %H:%M:%S')
            dates.append(ConvertTime(time).utc_to_local())
            values.append(i['main']['temp'])
        return pd.DataFrame(values, index=dates, columns=['values'])


#TODO create interpolation class here


class ConvertTime(object):
    """Handles GMT to local conversions"""

    def __init__(self, time):
        """:param time: a datetime object"""
        self.time = time

    def utc_to_local(self):
        """converts from gmt to local"""
        epoch = time.mktime(self.time.timetuple())
        offset = datetime.fromtimestamp(
            epoch) - datetime.utcfromtimestamp(epoch)
        return self.time + offset

    def local_to_utc(self):
        """converts from local to gmt"""
        epoch = time.mktime(self.time.timetuple())
        offset = datetime.fromtimestamp(
            epoch) - datetime.utcfromtimestamp(epoch)
        return self.time - offset
