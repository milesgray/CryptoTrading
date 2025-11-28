import requests
from free_proxy import FreeProxy

IP_GEO_API_KEY = "32a0be2e36fb4f748c33011292f3b0b9"
good_countries = ['BD', 'AS', 'BA', 'BW', 'KH', 'CD', 'CU', 'HK', 'KR', 'FI']
def get_proxy(country_id=good_countries, verbose=True):
    try:
        return {
            'http': FreeProxy(country_id=country_id, rand=True).get(),
            'https': FreeProxy(country_id=country_id, rand=True, https=True).get()
        }
    except Exception as e:
        if verbose: print(e)
        return None
def test_proxy(proxy, verbose=True):
    try:
        url = 'https://httpbin.org/ip'
        response = requests.get(url, proxies=proxy)
        result = response.json()
        if verbose: print(result)
        country_code = geo_lookup(result['origin'])
        return True
    except Exception as e:
        if verbose: print(e)
        return False
def geo_lookup(ip=None, verbose=True):
    url = f'https://api.ipgeolocation.io/ipgeo?apiKey={IP_GEO_API_KEY}'
    if ip: url = f"{url}&ip={ip}"
    response = requests.get(url)
    result = response.json()
    if verbose: print(f"Geo lookup result:\n{result}")
    return result["country_code2"]