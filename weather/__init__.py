import requests
from datetime import datetime

API_KEY = '7ae7f2704cd63c9a54e4432bf42d200f'
API_KEY = '44db6a862fba0b067b1930da0d769e98'

class Weather():
    def __init__(self):
        self.SNOW_THRESHOLD_LITTLE = 0.2
        self.SNOW_THRESHOLD = 1
        self.SNOW_THRESHOLD_LOT = 6
        self.RAIN_THRESHOLD_LITTLE = 0.05
        self.RAIN_THRESHOLD = 0.2
        self.RAIN_THRESHOLD_LOT = 0.5
        self.TEMP_COLD = 35
        self.TEMP_FUCKING_COLD = 15
        self.TEMP_WARM = 60
        self.TEMP_HOT = 75
        self.TEMP_FUCKING_HOT = 90

        self.weather_data = []
        self.weather = {}
        self.day_min = None
        self.day_max = None
        self.day_avg = None
        self.rain_total = 0
        self.snow_total = 0

    def check_weather(self, tomorrow):
        # request weather data from API
        params = {
            'id': '5128581',
            'mode': 'json',
            'units': 'imperial',
            'APPID': API_KEY
        }
        r = requests.get('http://api.openweathermap.org/data/2.5/forecast', params=params)
        data = r.json()

        # filter according to waking hours
        now = datetime.now()
        today = now.day
        if tomorrow: today += 1

        # parse weather data
        weather_data = []
        temps = []
        weather = {}
        snow_total = 0
        rain_total = 0
        for entry in data['list']:
            date = datetime.fromtimestamp(entry['dt'])
            if date.day == today and date.hour >= 9 and date.hour < 23:
                weather_data += [{
                    'date': datetime.fromtimestamp(entry['dt']),
                    'temp': entry['main']['temp'],
                    'min_temp': entry['main']['temp_min'],
                    'max_temp': entry['main']['temp_max'],
                    'snow_volume': entry['snow'],
                    'weather': entry['weather'][0]['main']
                }]

                temps += [(entry['main']['temp'], entry['main']['temp_min'], entry['main']['temp_max'])]
                weather_desc = entry['weather'][0]['main']
                if weather_desc not in weather:
                    weather[weather_desc] = 0
                weather[weather_desc] += 1

                if 'snow' in entry and entry['snow']: snow_total += entry['snow']['3h']
                if 'rain' in entry and entry['rain']: rain_total += entry['rain']['3h']

        if not weather_data:
            return

        # set weather data
        day_min = min(temps, key=lambda (a, min_t, b): min_t)
        day_max = max(temps, key=lambda (a, b, max_t): max_t)
        day_avg = sum([t[0] for t in temps])/len(temps)
        self.weather_data = weather_data
        self.weather = weather
        self.day_min = day_min
        self.day_max = day_max
        self.day_avg = day_avg
        self.rain_total = rain_total
        self.snow_total = snow_total

    def weather_statement(self, tomorrow):
        if not self.day_avg or not self.day_min or not self.day_max or not self.weather_data or not self.weather:
            return None

        result = '%s is ' % ('Tomorrow' if tomorrow else 'Today')
        # interpret temperature
        if self.day_avg < self.TEMP_FUCKING_COLD:
            result += 'fucking cold.'
        elif self.day_avg < self.TEMP_COLD:
            result += 'cold.'
        elif self.day_avg > self.TEMP_FUCKING_HOT:
            result += 'fucking hot.'
        elif self.day_avg > self.TEMP_HOT:
            result += 'hot.'
        elif self.day_avg > self.TEMP_WARM:
            result += 'warm.'
        else:
            result += '%f degrees.' % self.day_avg

        # interpret rain
        if 'Rain' in self.weather:
            result += ' It will rain'
            if self.rain_total > self.RAIN_THRESHOLD_LOT or\
                self.weather['Rain'] > 1 and self.rain_total > self.RAIN_THRESHOLD or\
                self.weather['Rain'] > 2 and self.rain_total > self.RAIN_THRESHOLD_LITTLE:
                result += ' a lot.'
            elif self.rain_total > self.RAIN_THRESHOLD or\
                self.weather['Rain'] > 1 and self.rain_total > self.RAIN_THRESHOLD_LITTLE:
                result += '.'
            elif self.rain_total > self.RAIN_THRESHOLD_LITTLE:
                result += ' a little.'

        # interpret snow
        if 'Snow' in self.weather:
            result += ' It will snow'
            if self.snow_total > self.SNOW_THRESHOLD_LOT or\
                self.weather['Snow'] > 1 and self.snow_total > self.SNOW_THRESHOLD or\
                self.weather['Snow'] > 2 and self.snow_total > self.SNOW_THRESHOLD_LITTLE:
                result += ' a lot.'
            elif self.snow_total > self.SNOW_THRESHOLD or\
                self.weather['Snow'] > 1 and self.snow_total > self.SNOW_THRESHOLD_LITTLE:
                result += '.'
            elif self.snow_total > self.SNOW_THRESHOLD_LITTLE:
                result += ' a little.'

        return result

    def get_weather(self, tomorrow):
        self.check_weather(tomorrow)
        return self.weather_statement(tomorrow)

if __name__ == '__main__':
    w = Weather()
    w.get_weather()
    print w.weather_statement()
