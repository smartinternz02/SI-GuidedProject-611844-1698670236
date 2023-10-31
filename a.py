import requests
from bs4 import BeautifulSoup

while(True):
    find=input()
    url=f"https://www.google.com/search?q={find}"
    # url=f"https://en.wikipedia.org/wiki/{find}"

    req=requests.get(url)
    soup=BeautifulSoup(req.text,"html.parser")

    # print(req.text)
    mysearch=soup.find("div",class_="BNeawe").text
    # mysearch=soup.find("div",class_="mw-parser-output").text
    print(mysearch)