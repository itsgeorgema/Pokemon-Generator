import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.serebii.net/pokemon/nationalpokedex.shtml"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())#[:1000])
#rows = soup.select('tbody tr')

data= []

#for row in rows:
#    d = dict()
    
#  d['name'] = row.select_one('.source-title').text.strip()
    
#    data.append(d)