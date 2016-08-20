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

    def __init__(self, api_key, series, start=None, end=None):
        """
        Create eia_api object and related attriobutes from json
        :param api_key: an API key that is provided by EIA
        :param start: a start date must be in the same date format as eia series
        :param end:a date
        :param *args: The series id (also called source key) is a
                       case-insensitive string consisting of letters, numbers, dashes
                       ("-") and periods (".") that uniquely identifies an EIA series\
                       multiple series can be submitted by comma separation ex: api_key, s1, s2
        """
        self.api_key = api_key
        self.series_id = series
        # self.series_id = [";".join(args)] TODO setup to handle *args
        self.start = start
        self.end = end
        self.json = self.get_series()
        self.dataframe = self.create_dataframes()

    def get_series(self):
        """
        Calls the EIA API with supplied api_key on init and series_id and return json
        """
        # default api params required for call
        api_parms = (
            ('api_key', self.api_key),
            ('series_id', self.series_id),
        )
        # add optional time parms
        if self.start and self.end is not None:
            api_parms = api_parms + (('start', self.start), ('end', self.end),)
        eia_req = requests.get(self.eia_url, params=api_parms)
        return json.loads(eia_req.text)

    def create_dataframes(self):
        """
        Creates a pandas dataframe of data key in json returned from get_series
        :param json: is an eia json object
        """
        df_dict = {}
        for series in self.json['series']:
            df = pd.DataFrame(self.get_values(series), index=self.get_dates(series),
                              columns=['values'])
            df_dict[series['series_id']] = df
        return df_dict

    @staticmethod
    def get_dates(series):
        """Parse dates from eia json['series']
        :param series: a series object returned by eia json['series']
        """
        # create a dict to look up datetime frequency values
        freq = {'A': '%Y', 'M': '%Y%m', 'W': '%Y%m%d',
                'D': '%Y%m%d', 'H': '%Y%m%d %H'}
        date_list = []
        for x in series['data']:
            # need to add this ugly bit to remove hourly time format from EIA
            time = x[0].replace('T', ' ')
            time = time.replace('Z', '')
            date_list.append(datetime.strptime(
                time, freq[series['f']]).strftime('%Y-%m-%d %H:%M:%S'))
        return date_list

    @staticmethod
    def get_values(series):
        """Parse values from eia json['series']
        :param series: a series object returned by eia json['series']
        """
        return [value[1] for value in series['data']]


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
        self.dataframe = self.create_dataframes(self.json)
        self.hourly_time = self.get_time_hourly(self.dataframe)
        self.hourly_temps = self.interpolate_weather(self.dataframe,
                                                     self.hourly_time, 'temp')

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

    def create_dataframes(self, json):
        """
        Creates a pandas dataframe of data key in json returned from get_series
        :param json: is an owm forecast json object
        """
        time_dict = {}
        # first loop through each 3 hr interval in forecast
        for i in json['list']:
            time = self.utc2local(datetime.strptime(
                i['dt_txt'], '%Y-%m-%d %H:%M:%S'))
            temps_dict = {'temp': None, 'temp_max': None, 'temp_min': None}
            # create nested dict for each temp attribute
            for j in temps_dict:
                temps_dict[j] = i['main'][j]
            time_dict[time] = temps_dict
        return pd.DataFrame.from_dict(time_dict, orient='index')

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

    @staticmethod
    def utc2local(utc):
        """
        Convert UTC time to local time
        :param utc: datetime object with UTC format
        """
        epoch = time.mktime(utc.timetuple())
        offset = datetime.fromtimestamp(
            epoch) - datetime.utcfromtimestamp(epoch)
        return utc + offset


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
